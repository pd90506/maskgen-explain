import os
import json

# Load environment configuration first
def load_env_config(json_path):
    """Load and set environment variables from config file"""
    with open(json_path, 'r') as file:
        env_config = json.load(file)
    os.environ['HF_HOME'] = env_config['HF_HOME']
    os.environ['HUGGINGFACE_HUB_TOKEN'] = env_config['access_token']

# Configuration dictionary
config = {
    # Environment settings
    'env': {
        'json_path': 'env_config.json',
        'results_dir': 'results',
    },
    # Model settings
    'model': {
        'pretrained_name': 'google/vit-base-patch16-224',
        'patch_size': 14,
        'output_dim': 1000,  # ImageNet classes
    },
}

# Set environment variables before importing transformers
load_env_config(config['env']['json_path'])

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    ViTModel,
    ViTConfig,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)

def load_models(device):
    """Initialize and load all required models"""
    pretrained_name = config['model']['pretrained_name']
    
    # Load configuration and processor
    model_config = ViTConfig.from_pretrained(pretrained_name)
    processor = ViTImageProcessor.from_pretrained(pretrained_name)
    
    # Initialize prediction model
    pred_model = ViTForImageClassification.from_pretrained(pretrained_name)
    pred_model.to(device)
    pred_model.eval()

    return processor, pred_model


class BaselineInfer():
    def __init__(self, ):
        super(BaselineInfer, self).__init__()
    
