import torch
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize
from transformers import ViTImageProcessor


def create_transforms(processor: ViTImageProcessor):
    """Create image transforms based on processor config."""
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    
    if "height" in processor.size:
        size = (processor.size["height"], processor.size["width"])
        crop_size = size
    elif "shortest_edge" in processor.size:
        size = processor.size["shortest_edge"]
        crop_size = (size, size)
    
    return Compose([
        RandomResizedCrop(crop_size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])

def get_preprocess(processor):
    """Apply transforms across a batch."""
    transforms = create_transforms(processor)
    def preprocess(example_batch):
        example_batch["pixel_values"] = [
            transforms(image.convert("RGB")) 
            for image in example_batch["image"]
        ]
        return example_batch
    return preprocess

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}