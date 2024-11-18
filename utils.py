import os
from PIL import Image
import numpy as np
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils import is_wandb_available
if is_wandb_available():
    import wandb
from transformers import PretrainedConfig

from modified_diffusers_modules.scheduling_ddim import DDIMScheduler

def save_model_card(
    repo_id: str,
    images=None,
    base_model=str,
    train_text_encoder=False,
    prompt=str,
    repo_folder=None,
    pipeline: DiffusionPipeline = None,
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# LoRA DreamBooth - {repo_id}

These are LoRA adaption weights for {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/). You can find some example images in the following. \n
{img_str}

LoRA for the text encoder was enabled: {train_text_encoder}.
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        prompt=prompt,
        model_description=model_description,
        inference=True,
    )
    tags = ["text-to-image", "diffusers", "lora", "diffusers-training"]
    if isinstance(pipeline, StableDiffusionPipeline):
        tags.extend(["stable-diffusion", "stable-diffusion-diffusers"])
    else:
        tags.extend(["if", "if-diffusers"])
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    logger,
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    generator,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    if args.seed is not None:
        generator.manual_seed(args.seed)
    # generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    if args.validation_images is None:
        images = []
        for _ in range(args.num_validation_images):
            with torch.cuda.amp.autocast():
                image = pipeline(**pipeline_args, generator=generator).images[0]
                images.append(image)
    else:
        images = []
        for image in args.validation_images:
            image = Image.open(image)
            with torch.cuda.amp.autocast():
                image = pipeline(**pipeline_args, image=image, generator=generator).images[0]
            images.append(image)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()

    return images


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds