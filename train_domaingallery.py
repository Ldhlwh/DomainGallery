
import argparse
import copy
import gc
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision as tv
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

from tqdm.auto import tqdm
from transformers import AutoTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# DomainGallery
from modified_diffusers_modules.unet_2d_condition import UNet2DConditionModel
from modified_diffusers_modules.scheduling_ddim import DDIMScheduler
from dataset import DomainGalleryDataset, PromptDataset, collate_fn
from utils import save_model_card, log_validation, import_model_class_from_model_name_or_path, tokenize_prompt, encode_prompt

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # DomainGallery stage 1: prior attribute erasure
    # or stage 2: finetuning
    parser.add_argument("--mode", type = str, required = True, choices = ['prior_attribute_erasure', 'finetuning'])

    # base model
    parser.add_argument("--pretrained_model_name_or_path", type = str, default = None, required = True, help = "Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type = str, default = None, required = False, help = "Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--variant", type = str, default = None, help = "Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",)
    parser.add_argument("--tokenizer_name", type = str, default = None, help = "Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--tokenizer_max_length", type = int, default = None, required = False, help = "The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.")
    parser.add_argument("--text_encoder_use_attention_mask", action = "store_true", required = False, help = "Whether to use attention mask for the text encoder")
    parser.add_argument("--class_labels_conditioning", required = False, default = None, help = "The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.")
    
    # data
    parser.add_argument("--resolution", type = int, default = 512, help = "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution")
    parser.add_argument("--center_crop", default = False, action = "store_true", help = "Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping.")
    # target data
    parser.add_argument("--target_data_dir", type = str, default = None, help = "A folder containing the training data of target domain images.")
    parser.add_argument("--target_prompt", type = str, default = None, help = "The prompt with identifier specifying the domain")
    # source (class, prior) data
    parser.add_argument("--source_data_dir", type = str, default = None, help = "A folder containing the training data of source images.") 
    parser.add_argument("--source_prompt", type = str, default = None, help = "The prompt to specify source images.")
    parser.add_argument("--num_source_images", type = int, default = 100, help = "Minimal source images for prior preservation loss. If there are not enough images already present in source_data_dir, additional images will be sampled with source_prompt.")
    parser.add_argument("--sample_batch_size", type = int, default = 4, help = "Batch size (per device) for sampling images.")
    parser.add_argument("--source_generation_precision", type = str, default = None, choices = ["no", "fp32", "fp16", "bf16"], help = "Choose source generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32.")

    # DomainGallery
    parser.add_argument("--rank", type = int, default = 4, help = "The dimension of the LoRA update matrices.")
    parser.add_argument("--pretrained_lora_path", type = str, default = None, help = "The path of the LoRA module (where the prior attributes are erased) to start with for stage 2")
    parser.add_argument("--prior_loss_weight", type = float, default = 1.0, help = "The weight of prior preservation loss.")
    parser.add_argument("--erasure_loss_weight", type = float, default = 10.0, help = "The weight of prior attribute erasure loss (Eq. 2)")
    parser.add_argument("--disen_loss_weight", type = float, default = 10.0, help = "The weight of attribute disentanglement loss (Eq. 3)")
    parser.add_argument("--sim_loss_weight", type = float, default = 1.0, help = "The weight of similarity consistency loss (Eq. 4)")
    parser.add_argument("--num_denoise_steps", type = int, default = 5)
    parser.add_argument("--denoise_guidance_scale", type = float, default = 1.0)
    parser.add_argument("--offset_noise_scale", type = float, default = 0.0)

    # training
    # basic
    parser.add_argument("--output_dir", type = str, default = "runs", help = "The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--seed", type = int, default = None, help = "A seed for reproducible training.")
    parser.add_argument("--train_text_encoder", action = "store_true", help = "Whether to train the text encoder. If set, the text encoder should be float32 precision.")
    parser.add_argument("--train_batch_size", type = int, default = 4, help = "Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type = int, default = 1)
    parser.add_argument("--max_train_steps", type = int, default = None, help = "Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument("--resume_from_checkpoint", type = str, default = None, help = "Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `latest` to automatically select the last available checkpoint.")
    parser.add_argument("--mixed_precision", type = str, default = None, choices = ["no", "fp16", "bf16"], help = "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.")
    # optimization
    parser.add_argument("--learning_rate", type = float, default = 5e-4, help = "Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action = "store_true", default = False, help = "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type = str, default = "constant", help = 'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]')
    parser.add_argument("--lr_warmup_steps", type = int, default = 500, help = "Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type = int, default = 1, help = "Number of hard resets of the lr in cosine_with_restarts scheduler.")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--adam_beta1", type = float, default = 0.9, help = "The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type = float, default = 0.999, help = "The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type = float, default = 1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type = float, default = 1e-08, help = "Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default = 1.0, type = float, help = "Max gradient norm.")
    # dataset
    parser.add_argument("--dataloader_num_workers", type = int, default = 0, help = "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    # saving VRAM
    parser.add_argument("--use_8bit_adam", action = "store_true", help = "Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--gradient_accumulation_steps", type = int, default = 1, help = "Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action = "store_true", help = "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action = "store_true", help = "Whether or not to use xformers.")
    parser.add_argument("--pre_compute_text_embeddings", action = "store_true", help = "Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`.")

    # checkpointing & validation
    parser.add_argument("--checkpointing_steps", type = int, default = 500, help = "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final checkpoints in case they are better than the last checkpoint, and are also suitable for resuming training using `--resume_from_checkpoint`.")
    parser.add_argument("--checkpoints_total_limit", type = int, default = None, help = "Max number of checkpoints to store.")
    parser.add_argument("--validation_prompt", type = str, default = None, help = "A prompt that is used during validation to verify that the model is learning.")
    parser.add_argument("--num_validation_images", type = int, default = 4, help = "Number of images that should be generated during validation with `validation_prompt`.")
    parser.add_argument("--validation_steps", type = int, default = 100, help = "Run validation every X steps")
    parser.add_argument('--validation_guidance_scale', type = float, default = 7.5)
    parser.add_argument('--validation_inference_steps', type = int, default = 50)
    parser.add_argument("--start_checkpointing_steps", type = int, default = 0)
    parser.add_argument("--start_validation_steps", type = int, default = 0)
    parser.add_argument("--validation_images", required = False, default = None, nargs = "+", help = "Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.")
    
    # misc
    parser.add_argument("--push_to_hub", action = "store_true", help = "Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type = str, default = None, help = "The token to use to push to the Model Hub.")
    parser.add_argument("--hub_model_id", type = str, default = None, help = "The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--logging_dir", type = str, default = "logs", help = "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    parser.add_argument("--allow_tf32", action = "store_true", help = "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices")
    parser.add_argument("--report_to", type = str, default = "tensorboard", help = 'The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.')
    parser.add_argument("--local_rank", type = int, default = -1, help = "For distributed training: local_rank")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.source_data_dir is None:
        raise ValueError("You must specify a source data directory.")
    if args.mode == 'prior_attribute_erasure':
        if args.target_data_dir is not None:
            warnings.warn(f"You need not use --target_data_dir for mode {args.mode}.")
        if args.target_prompt is None and args.erasure_loss_weight > 0.0:
            raise ValueError(f"You must specify a target prompt if erasing prior attributes.")
    if args.mode == 'finetuning' and args.target_data_dir is None:
        raise ValueError(f"You must specify a target data directory for mode {args.mode}.")

    if args.train_text_encoder and args.pre_compute_text_embeddings:
        raise ValueError("`--train_text_encoder` cannot be used with `--pre_compute_text_embeddings`")
    if args.train_text_encoder:
        warnings.warn(f'Training text encoder was not tested for DomainGallery. Use it at your own risk.')

    if args.mode == 'finetuning' and args.pretrained_lora_path is None:
        warnings.warn(f'You did not specify the pretrained LoRA module where the prior attributes have been pre-erased. Will start from default initialization.')

    if args.validation_prompt is None:
        args.validation_prompt = args.target_prompt

    return args


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        mixed_precision = args.mixed_precision,
        log_with = args.report_to,
        project_config = accelerator_project_config,
    )

    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (sayakpaul): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError("Gradient accumulation is not supported when training the text encoder in distributed training. Please set gradient_accumulation_steps to 1. This feature will be supported in the future.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate source images.
    source_images_dir = Path(args.source_data_dir)
    if not source_images_dir.exists():
        source_images_dir.mkdir(parents = True)
    cur_source_images = len(list(source_images_dir.iterdir()))

    if cur_source_images < args.num_source_images:
        torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
        if args.source_generation_precision == "fp32":
            torch_dtype = torch.float32
        elif args.source_generation_precision == "fp16":
            torch_dtype = torch.float16
        elif args.source_generation_precision == "bf16":
            torch_dtype = torch.bfloat16
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype = torch_dtype,
            safety_checker = None,
            revision = args.revision,
            variant = args.variant,
        )
        pipeline.set_progress_bar_config(disable = True)

        num_new_images = args.num_source_images - cur_source_images
        logger.info(f"Number of source images to sample: {num_new_images}.")

        sample_dataset = PromptDataset(args.source_prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size = args.sample_batch_size)
        sample_dataloader = accelerator.prepare(sample_dataloader)
        pipeline.to(accelerator.device)

        for example in tqdm(sample_dataloader, desc=f"Generating source images using prompt {args.source_prompt}", disable = not accelerator.is_local_main_process):
            images = pipeline(example["prompt"]).images
            for i, image in enumerate(images):
                hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = source_images_dir / f"{example['index'][i] + cur_source_images}-{hash_image}.jpg"
                image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant)

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning("xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details.")
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # now we will add new LoRA weights to the attention layers
    if args.pretrained_lora_path is None:
        unet_lora_config = LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"])
        unet.add_adapter(unet_lora_config)
        if args.train_text_encoder:
            text_lora_config = LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian", target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
            text_encoder.add_adapter(text_lora_config) 
    else:
        lora_state_dict, lora_network_alphas = LoraLoaderMixin.lora_state_dict(args.pretrained_lora_path)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas = lora_network_alphas, unet = unet, adapter_name = 'default')
        unet._hf_peft_config_loaded = True
        logger.info(f'LoRA weights at {args.pretrained_lora_path} have been loaded into unet')
        if args.train_text_encoder:
            LoraLoaderMixin.load_lora_into_text_encoder(lora_state_dict, network_alphas = lora_network_alphas, text_encoder = text_encoder, lora_scale = LoraLoaderMixin.lora_scale, adapter_name = 'default')
            text_encoder._hf_peft_config_loaded = True
            logger.info(f'LoRA weights at {args.pretrained_lora_path} have been loaded into text_encoder')        

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                elif isinstance(model, type(unwrap_model(text_encoder))):
                    text_encoder_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                # elif isinstance(model, clip.model.CLIP) or isinstance(model, type(unwrap_model(vae))):
                #     continue
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            LoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder))):
                text_encoder_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.train_text_encoder:
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_)

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [unet_]
            if args.train_text_encoder:
                models.append(text_encoder_)

            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        if args.train_text_encoder:
            models.append(text_encoder)

        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if args.train_text_encoder:
        params_to_optimize = params_to_optimize + list(filter(lambda p: p.requires_grad, text_encoder.parameters()))

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    def compute_text_embeddings(prompt):
        with torch.no_grad():
            text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=args.tokenizer_max_length)
            prompt_embeds = encode_prompt(
                text_encoder,
                text_inputs.input_ids,
                text_inputs.attention_mask,
                text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
            )
        return prompt_embeds

    if args.pre_compute_text_embeddings:
        pre_computed_target_prompt_encoder_hidden_states = compute_text_embeddings(args.target_prompt)
        pre_computed_source_prompt_encoder_hidden_states = compute_text_embeddings(args.source_prompt) if args.source_prompt is not None else None
        pre_computed_uncond_encoder_hidden_states = compute_text_embeddings("")
        validation_prompt_encoder_hidden_states = compute_text_embeddings(args.validation_prompt) if args.validation_prompt is not None else None
        validation_prompt_negative_prompt_embeds = compute_text_embeddings("")

        text_encoder = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
    else:
        pre_computed_target_prompt_encoder_hidden_states = None
        pre_computed_source_prompt_encoder_hidden_states = None
        pre_computed_uncond_encoder_hidden_states = None
        validation_prompt_encoder_hidden_states = None
        validation_prompt_negative_prompt_embeds = None

    # Dataset and DataLoaders creation:
    train_dataset = DomainGalleryDataset(tokenizer = tokenizer,
        target_data_root = args.target_data_dir, target_prompt = args.target_prompt, target_prompt_encoder_hidden_states = pre_computed_target_prompt_encoder_hidden_states,
        source_data_root = args.source_data_dir, source_prompt=args.source_prompt, source_num = args.num_source_images, source_prompt_encoder_hidden_states=pre_computed_source_prompt_encoder_hidden_states,
        size = args.resolution, center_crop = args.center_crop, tokenizer_max_length = args.tokenizer_max_length,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.train_batch_size,
        shuffle = True,
        collate_fn = collate_fn,
        num_workers = args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer = optimizer,
        num_warmup_steps = args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps = args.max_train_steps * accelerator.num_processes,
        num_cycles = args.lr_num_cycles,
        power = args.lr_power,
    )

    # Extra modules and functions for stage 2: finetuning.
    if args.mode == 'finetuning':
        pred_noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder = "scheduler").to(accelerator.device)

        def get_pred_timesteps(start_timestep, num_steps):
            if start_timestep <= num_steps - 1:
                return list(range(start_timestep, -1, -1))
            else:
                return torch.linspace(start_timestep, 0, num_steps + 1).round().int().tolist()[:-1]
            
        def pairwise_similarity_matrix(feat):
            sim = torch.nn.functional.cosine_similarity
            num = feat.shape[0]
            feat = feat.reshape(num, -1)
            cos_mat = sim(feat.unsqueeze(1), feat.unsqueeze(0), dim = -1)
            matrix = cos_mat.triu(diagonal = 1)[:, 1:] + cos_mat.tril(diagonal = -1)[:, :-1]
            return matrix
        
        def pairwise_similarity_loss(target, source):
            sfm = torch.nn.Softmax(dim = 1)
            kl_loss = torch.nn.KLDivLoss()
            matrix_target = sfm(pairwise_similarity_matrix(target))
            matrix_source = sfm(pairwise_similarity_matrix(source))
            return kl_loss(torch.log(matrix_target), matrix_source)
        
        def recurrent_denoise(unet, scheduler, noisy_model_input, t_list, encoder_hidden_states, class_labels, 
                              guidance_scale = 1.0, encoder_hidden_states_uncond = None):
            pred_prev = noisy_model_input.clone()

            for i in range(len(t_list)):
                cur_t = t_list[i]
                prev_t = -1 if i == len(t_list) - 1 else t_list[i + 1]

                if guidance_scale == 1.0:
                    cur_model_pred = unet(pred_prev, cur_t, encoder_hidden_states, class_labels = class_labels, return_dict = False)[0]
                else:   # guidance_scale > 1.0
                    input_pred_prev = torch.cat([pred_prev, pred_prev], dim = 0)
                    input_encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_uncond.repeat(encoder_hidden_states.shape[0], 1, 1)], dim = 0)
                    cur_model_pred = unet(input_pred_prev, cur_t, input_encoder_hidden_states, class_labels = class_labels, return_dict = False)[0]
                    model_pred, model_pred_uncond = torch.chunk(cur_model_pred, 2, dim = 0)
                    cur_model_pred = model_pred_uncond + guidance_scale * (model_pred - model_pred_uncond)

                cur_pred = scheduler.step(cur_model_pred, cur_t, pred_prev, prev_timestep = prev_t)
                pred_prev, pred_ori = cur_pred.prev_sample, cur_pred.pred_original_sample

            return pred_ori


    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, text_encoder, optimizer, train_dataloader, lr_scheduler)
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        tracker_config.pop("validation_images")
        accelerator.init_trackers("domaingallery", config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(range(0, args.max_train_steps), initial = initial_global_step, desc = "Steps", disable = not accelerator.is_local_main_process,)
    torch_generator = torch.Generator(device = accelerator.device)

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor

                # Sample noise that we'll add to the model input
                if args.offset_noise_scale > 0.0:
                    noise = torch.randn_like(model_input) + args.offset_noise_scale * torch.randn(model_input.shape[0], model_input.shape[1], 1, 1, device = model_input.device)
                else:
                    noise = torch.randn_like(model_input)

                bsz, channels, _, _ = model_input.shape

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device = model_input.device).repeat(bsz)

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # Get the text embedding for conditioning
                if args.pre_compute_text_embeddings:
                    encoder_hidden_states = batch["input_ids"]
                    encoder_hidden_states_uncond = pre_computed_uncond_encoder_hidden_states
                else:
                    encoder_hidden_states = encode_prompt(text_encoder, batch["input_ids"], batch["attention_mask"], text_encoder_use_attention_mask = args.text_encoder_use_attention_mask)
                    encoder_hidden_states_uncond = compute_text_embeddings("")

                if unwrap_model(unet).config.in_channels == channels * 2:
                    noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                class_labels = timesteps if args.class_labels_conditioning == "timesteps" else None

                # Predict the noise residual
                model_pred = unet(noisy_model_input, timesteps, encoder_hidden_states, class_labels = class_labels, return_dict = False)[0]

                # if model predicts variance, throw away the prediction. we will only train on the
                # simplified training objective. This means that all schedulers using the fine tuned
                # model must be configured to use one of the fixed variance variance types.
                if model_pred.shape[1] == 6:
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                # Get the source for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                # preparation
                if args.mode == 'prior_attribute_erasure':
                    # source images only
                    model_pred_src = model_pred
                    target_src = target
                    noisy_model_input_src = noisy_model_input
                    encoder_hidden_states_src = encoder_hidden_states
                    if args.pre_compute_text_embeddings:
                        encoder_hidden_states_tgt = pre_computed_target_prompt_encoder_hidden_states.repeat(bsz, 1, 1)
                    else:
                        encoder_hidden_states_tgt = compute_text_embeddings(args.target_prompt).repeat(bsz, 1, 1)
                    timesteps_src = timesteps
                elif args.mode == 'finetuning':
                    model_pred_tgt, model_pred_src = torch.chunk(model_pred, 2, dim = 0)
                    target_tgt, target_src = torch.chunk(target, 2, dim = 0)
                    _, noisy_model_input_src = torch.chunk(noisy_model_input, 2, dim = 0)
                    encoder_hidden_states_tgt, encoder_hidden_states_src = torch.chunk(encoder_hidden_states, 2, dim = 0)
                    _, timesteps_src = torch.chunk(timesteps, 2, dim = 0)

                # target loss & prior preservation loss
                if args.mode == 'prior_attribute_erasure':
                    prior_loss = F.mse_loss(model_pred_src.float(), target_src.float(), reduction = "mean")
                    loss = args.prior_loss_weight * prior_loss
                elif args.mode == 'finetuning':
                    target_loss = F.mse_loss(model_pred_tgt.float(), target_tgt.float(), reduction = "mean")
                    prior_loss = F.mse_loss(model_pred_src.float(), target_src.float(), reduction = "mean")
                    loss = target_loss + args.prior_loss_weight * prior_loss

                # prior attribute erasure loss (prior_attribute_erasure only)
                if args.mode == 'prior_attribute_erasure' and args.erasure_loss_weight > 0.0:
                    model_pred_s2t = unet(noisy_model_input_src, timesteps_src, encoder_hidden_states_tgt, class_labels = class_labels, return_dict = False)[0]
                    erasure_loss = F.mse_loss(model_pred_s2t.float(), model_pred_src.float().detach(), reduction = "mean")
                    loss = loss + args.erasure_loss_weight * erasure_loss
                
                # attribute disentanglement loss (both prior_attribute_erasure & finetuning)
                if args.disen_loss_weight > 0.0:
                    with torch.no_grad():
                        unet.disable_adapters()
                        model_pred_src_wo_lora = unet(noisy_model_input_src, timesteps_src, encoder_hidden_states_src, class_labels = class_labels, return_dict = False)[0]
                        unet.enable_adapters()
                    disen_loss = F.mse_loss(model_pred_src.float(), model_pred_src_wo_lora.float().detach(), reduction = "mean")
                    loss = loss + args.disen_loss_weight * disen_loss

                # similarity consistency loss (finetuning only)
                if args.mode == 'finetuning' and args.sim_loss_weight > 0.0:
                    pred_timesteps = get_pred_timesteps(timesteps[0], num_steps = args.num_denoise_steps)

                    # noisy source latent -> denoised target latent
                    pred_s2t = recurrent_denoise(unet, pred_noise_scheduler, noisy_model_input_src, pred_timesteps, encoder_hidden_states_tgt, class_labels, 
                                                    encoder_hidden_states_uncond = encoder_hidden_states_uncond,
                                                    guidance_scale = args.denoise_guidance_scale)
                    embed_s2t = unet.encode_embedding(pred_s2t, 0, encoder_hidden_states_tgt, class_labels = class_labels, return_dict = False)
                        
                    # noisy source latent -> denoised source latent
                    with torch.no_grad():
                        pred_src = recurrent_denoise(unet, pred_noise_scheduler, noisy_model_input_src, pred_timesteps, encoder_hidden_states_src, class_labels,
                                                        encoder_hidden_states_uncond = encoder_hidden_states_uncond,
                                                        guidance_scale = args.denoise_guidance_scale)
                        embed_src = unet.encode_embedding(pred_src, 0, encoder_hidden_states_src, class_labels = class_labels, return_dict = False)

                    sim_loss = sum([pairwise_similarity_loss(embed_s2t[i], embed_src[i].detach()) for i in range(len(embed_s2t))]) / len(embed_s2t)
                    loss = loss + args.sim_loss_weight * sim_loss
                    
                # backward & update
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step >= args.start_checkpointing_steps and global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and (global_step == 1 or (global_step >= args.start_validation_steps and global_step % args.validation_steps == 0)):
                        pipeline = DiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet = unwrap_model(unet),
                            text_encoder = None if args.pre_compute_text_embeddings else unwrap_model(text_encoder),
                            revision = args.revision, variant = args.variant, torch_dtype = weight_dtype,
                        )

                        if args.pre_compute_text_embeddings:
                            pipeline_args = {
                                "prompt_embeds": validation_prompt_encoder_hidden_states,
                                "negative_prompt_embeds": validation_prompt_negative_prompt_embeds,
                            }
                        else:
                            pipeline_args = {"prompt": args.validation_prompt}
                        pipeline_args["guidance_scale"] = args.validation_guidance_scale
                        pipeline_args["num_inference_steps"] = args.validation_inference_steps
                            
                        images = log_validation(logger, pipeline, args, accelerator, pipeline_args, epoch, generator = torch_generator)
                        os.makedirs(os.path.join(args.output_dir, 'val_images'), exist_ok = True)
                        canvas = [tv.transforms.functional.pil_to_tensor(image).unsqueeze(0) for image in images]
                        canvas = tv.utils.make_grid(torch.cat(canvas), nrow = len(canvas))
                        tv.io.write_jpeg(canvas, os.path.join(args.output_dir, 'val_images', f'checkpoint-{global_step}.jpg'))

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            logs['prior_loss'] = prior_loss.detach().item()
            if args.mode == 'finetuning':
                logs['target_loss'] = target_loss.detach().item()
            if args.mode == 'prior_attribute_erasure' and args.erasure_loss_weight > 0.0:
                logs['erasure_loss'] = erasure_loss.detach().item()
            if args.disen_loss_weight > 0.0:
                logs['disen_loss'] = disen_loss.detach().item()
            if args.mode == 'finetuning' and args.sim_loss_weight > 0.0:
                logs['sim_loss'] = sim_loss.detach().item()

            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        unet = unet.to(torch.float32)

        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        if args.train_text_encoder:
            text_encoder = unwrap_model(text_encoder)
            text_encoder_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder))
        else:
            text_encoder_state_dict = None

        LoraLoaderMixin.save_lora_weights(
            save_directory = args.output_dir,
            unet_lora_layers = unet_lora_state_dict,
            text_encoder_lora_layers = text_encoder_state_dict,
        )

        # Final inference
        # Load previous pipeline
        pipeline = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, revision = args.revision, variant = args.variant, torch_dtype = weight_dtype)
        pipeline.load_lora_weights(args.output_dir, weight_name = "pytorch_lora_weights.safetensors")

        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline_args = {"prompt": args.validation_prompt, "guidance_scale": args.validation_guidance_scale, "num_inference_steps": args.validation_inference_steps}
            images = log_validation(logger, pipeline, args, accelerator, pipeline_args, epoch, generator = torch_generator, is_final_validation=True)

        if args.push_to_hub:
            save_model_card(repo_id, images=images, base_model=args.pretrained_model_name_or_path, train_text_encoder=args.train_text_encoder, prompt=args.target_prompt, repo_folder=args.output_dir, pipeline=pipeline)
            upload_folder(repo_id=repo_id, folder_path=args.output_dir, commit_message="End of training", ignore_patterns=["step_*", "epoch_*"])

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)