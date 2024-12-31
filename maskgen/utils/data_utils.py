from datasets import load_dataset


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

