
from datasets import load_dataset

def load_imagenet_eval_data(seed):
    """Load and preprocess the dataset"""
    dataset = load_dataset("mrm8488/ImageNet1K-val")
    dataset = dataset['train']
    splits = dataset.train_test_split(test_size=0.1, seed=seed)
    test_ds = splits['test']
    splits = splits['train'].train_test_split(test_size=0.1, seed=seed)
    
    return splits['train'], splits['test'], test_ds

