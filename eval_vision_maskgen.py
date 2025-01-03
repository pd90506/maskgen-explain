from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel, ViTConfig
from maskgen.vision_models.vision_maskgen import MaskGeneratingModel, convert_to_peft
from maskgen.utils import get_preprocess, collate_fn, load_imagenet
from torch.utils.data import DataLoader
import torch
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Flatten config for easier access
    flat_config = {}
    flat_config.update(config['evaluation'])
    flat_config.update(config['model'])
    flat_config.update(config['dataset'])
    
    return flat_config

def plot_heatmap(mask: np.ndarray, image_id: int, save_path: str):
    """Plot and save heatmap visualization."""
    plt.figure(figsize=(6, 6))
    sns.heatmap(mask, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title(f'Attention Mask - Image {image_id}')
    plt.savefig(os.path.join(save_path, f'heatmap_{image_id}.png'))
    plt.close()

def main():
    # Load configuration
    config = load_config('eval_config.json')
    
    # Create results directory if it doesn't exist
    if not os.path.exists(config['results_path']):
        os.makedirs(config['results_path'])

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models and processor
    pretrained_name = config['pretrained_name']
    vit_config = ViTConfig.from_pretrained(pretrained_name)
    processor = ViTImageProcessor.from_pretrained(pretrained_name)
    
    # Target model for explanation
    target_model = ViTForImageClassification.from_pretrained(pretrained_name)
    target_model.eval()
    target_model.to(device)

    # Create and load maskgen model
    base_model = ViTModel.from_pretrained(pretrained_name)
    peft_model = convert_to_peft(base_model)
    maskgen_model = MaskGeneratingModel(
        base_model=peft_model,
        hidden_size=vit_config.hidden_size,
        num_classes=vit_config.num_labels,
        freeze_base=config['freeze_base']
    )
    
    # Load trained weights
    maskgen_model.load_state_dict(torch.load(config['model_path']))
    maskgen_model.eval()
    maskgen_model.to(device)

    # Data preprocessing
    dataset = load_imagenet(split='tiny', access_token=config['api_key'])
    preprocess = get_preprocess(processor)
    dataset.set_transform(preprocess)
    
    # Get dataloader
    eval_dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        collate_fn=collate_fn, 
        shuffle=False
    )

    # Evaluation loop
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            pixel_values = batch['pixel_values'].to(device)
            
            # Get model predictions
            pred_logits = target_model(pixel_values).logits
            pseudo_label = pred_logits.argmax(-1).unsqueeze(-1)
            
            # Get masks from maskgen model
            dist, _, _ = maskgen_model.get_dist_critic(pixel_values, pseudo_label)
            masks = dist.probs  # Shape: [batch_size, 196]
            
            # Reshape masks to 14x14
            masks = masks.reshape(-1, 14, 14).cpu().numpy()
            
            # Plot and save heatmaps
            for i, mask in enumerate(masks):
                image_id = batch_idx * config['batch_size'] + i
                plot_heatmap(mask, image_id, config['results_path'])
                
            # Only process the first few batches if specified
            if config['max_samples'] and (batch_idx + 1) * config['batch_size'] >= config['max_samples']:
                break

if __name__ == "__main__":
    main()
