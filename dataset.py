from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from PIL import Image
from utils import tokenize_prompt


class DomainGalleryDataset(Dataset):
    """
    A dataset to prepare the target and/or source images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(
        self,
        tokenizer,
        
        # provide target data in mode 'finetuning'
        target_data_root = None,
        target_prompt = None,
        target_prompt_encoder_hidden_states = None,
        # provide source data in both modes
        source_data_root = None,
        source_prompt = None,
        source_num = None,
        source_prompt_encoder_hidden_states = None,
        
        size = 512,
        center_crop = False,
        tokenizer_max_length = None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.target_prompt_encoder_hidden_states = target_prompt_encoder_hidden_states
        self.source_prompt_encoder_hidden_states = source_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length
        self._length = 0

        if target_data_root is None and source_data_root is None:
            raise ValueError(f'Target/source data cannot be None at the same time')

        if target_data_root is not None:
            self.target_data_root = Path(target_data_root)
            self.target_images_path = list(self.target_data_root.iterdir())
            self.num_target_images = len(self.target_images_path)
            self.target_prompt = target_prompt
            self._length = self.num_target_images
        else:
            self.target_data_root = None

        if source_data_root is not None:
            self.source_data_root = Path(source_data_root)
            self.source_data_root.mkdir(parents = True, exist_ok = True)
            self.source_images_path = list(self.source_data_root.iterdir())
            if source_num is not None:
                self.num_source_images = min(len(self.source_images_path), source_num)
            else:
                self.num_source_images = len(self.source_images_path)
            self._length = max(self.num_source_images, self._length)
            self.source_prompt = source_prompt
        else:
            self.source_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        if self.target_data_root:
            target_image_index = index % self.num_target_images
            target_image = Image.open(self.target_images_path[target_image_index])
            target_image = exif_transpose(target_image)

            if not target_image.mode == "RGB":
                target_image = target_image.convert("RGB")
            example["target_images"] = self.image_transforms(target_image)
            example["target_image_index"] = target_image_index

            if self.target_prompt_encoder_hidden_states is not None:
                example["target_prompt_ids"] = self.target_prompt_encoder_hidden_states
            else:
                text_inputs = tokenize_prompt(self.tokenizer, self.target_prompt, tokenizer_max_length=self.tokenizer_max_length)
                example["target_prompt_ids"] = text_inputs.input_ids
                example["target_attention_mask"] = text_inputs.attention_mask

        if self.source_data_root:
            source_image_index = index % self.num_source_images
            source_image = Image.open(self.source_images_path[source_image_index])
            source_image = exif_transpose(source_image)

            if not source_image.mode == "RGB":
                source_image = source_image.convert("RGB")
            example["source_images"] = self.image_transforms(source_image)
            example["source_image_index"] = source_image_index

            if self.source_prompt_encoder_hidden_states is not None:
                example["source_prompt_ids"] = self.source_prompt_encoder_hidden_states
            else:
                source_text_inputs = tokenize_prompt(self.tokenizer, self.source_prompt, tokenizer_max_length=self.tokenizer_max_length)
                example["source_prompt_ids"] = source_text_inputs.input_ids
                example["source_attention_mask"] = source_text_inputs.attention_mask

        return example
    

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate source images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def collate_fn(examples):
    has_attention_mask = "target_attention_mask" in examples[0] or "source_attention_mask" in examples[0]
    has_target_image = 'target_images' in examples[0]
    has_source_image = 'source_images' in examples[0]

    batch = {}
    input_ids, pixel_values, attention_mask = [], [], []

    if has_target_image:
        input_ids += [example["target_prompt_ids"] for example in examples]
        pixel_values += [example["target_images"] for example in examples]
        if has_attention_mask:
            attention_mask += [example["target_attention_mask"] for example in examples]
        batch['target_image_indices'] = [example["target_image_index"] for example in examples]

    # Concat target and source examples to avoid doing two forward passes.
    if has_source_image:
        input_ids += [example["source_prompt_ids"] for example in examples]
        pixel_values += [example["source_images"] for example in examples]
        if has_attention_mask:
            attention_mask += [example["source_attention_mask"] for example in examples]
        batch['source_image_indices'] = [example["source_image_index"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    batch['pixel_values'] = pixel_values.to(memory_format=torch.contiguous_format).float()
    batch['input_ids'] = torch.cat(input_ids, dim=0)
    if has_attention_mask:
        batch["attention_mask"] = attention_mask

    return batch