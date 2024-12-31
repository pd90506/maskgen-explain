from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel, ViTConfig
from maskgen.vision_models.vision_maskgen import MaskGeneratingModel, convert_to_peft
from maskgen.utils import get_preprocess, collate_fn, load_imagenet
from maskgen.trainer import PPOTrainer
from torch.utils.data import DataLoader
import wandb
import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten config for easier access
    flat_config = {}
    flat_config.update(config['training'])
    flat_config.update(config['model'])
    flat_config.update({
        'wandb_project': config['wandb']['project']
    })
    
    return flat_config

def main():
    # Load configuration
    config = load_config('config.yaml')
    
    # Create save directory if it doesn't exist
    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])

    # Initialize wandb
    wandb.init(project=config['wandb_project'], config=config)

    # Load models and processor
    pretrained_name = config['pretrained_name']
    vit_config = ViTConfig.from_pretrained(pretrained_name)
    processor = ViTImageProcessor.from_pretrained(pretrained_name)
    
    # Target model for explanation
    target_model = ViTForImageClassification.from_pretrained(pretrained_name)
    target_model.eval()

    # Create maskgen model
    base_model = ViTModel.from_pretrained(pretrained_name)
    peft_model = convert_to_peft(base_model)
    maskgen_model = MaskGeneratingModel(
        base_model=peft_model,
        hidden_size=vit_config.hidden_size,
        num_classes=vit_config.num_labels,
        freeze_base=config['freeze_base']
    )

    # Data preprocessing
    dataset = load_imagenet(split='tiny')
    preprocess = get_preprocess(processor)
    dataset.set_transform(preprocess)
    
    # get dataloader
    train_dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        collate_fn=collate_fn, 
        shuffle=True
    )

    # Initialize PPO model
    ppo_trainer = PPOTrainer(
        maskgen_model=maskgen_model,
        target_model=target_model,
        config=config
    )

    # Start training
    ppo_trainer.train(train_dataloader)

    # close wandb
    wandb.finish()

if __name__ == "__main__":
    main()