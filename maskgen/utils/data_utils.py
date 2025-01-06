from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from maskgen.utils.image_utils import get_preprocess, collate_fn
import numpy as np

def load_imagenet(split='tiny', access_token=None):
    """Load and preprocess ImageNet dataset
    
    Args:
        split (str): One of 'train', 'val', 'test' or 'tiny'
        access_token (str): HuggingFace access token required for ImageNet access
        
    Returns:
        dataset split as requested
    """
    if split == 'tiny':
        dataset = load_dataset("mrm8488/ImageNet1K-val", split='train')
        return dataset

    else:
        if access_token is None:
            raise ValueError("Access token is required to load the official ImageNet dataset. "
                           "Please provide a valid HuggingFace access token.")
        dataset = load_dataset("imagenet-1k", token=access_token)
        if split == 'train':
            return dataset['train']
        elif split == 'val':
            return dataset['validation']
        elif split == 'test':
            return dataset['test']
        
    raise ValueError(f"Invalid split '{split}'. Must be one of: train, val, test, tiny")


def get_imagenet_dataloader(split, processor, batch_size, shuffle, access_token=None, num_samples=None):
    """Get DataLoader for ImageNet dataset
    
    Args:
        split (str): One of 'train', 'val', 'test' or 'tiny'
        access_token (str): HuggingFace access token required for ImageNet access
        processor: Image processor for ViT model
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle the dataset
        
    Returns:
        DataLoader for the requested dataset split
    """
    dataset = load_imagenet(split, access_token)
    preprocess = get_preprocess(processor)
    dataset.set_transform(preprocess)

    if num_samples is not None:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = Subset(dataset, indices)

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)