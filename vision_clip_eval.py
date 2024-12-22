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
        'results_dir': 'results/vision-clip-vit',
        # 'model_path': 'trained/skilled-snowball-47_vision_clip_20241216-221712.pt',
        'model_path': 'trained/spring-blaze-45_vision_clip_20241214-161824.pt',
    },
    
    # Model settings
    'model': {
        'pretrained_name': 'google/vit-base-patch16-224',
        'patch_size': 14,
        'output_dim': 1000,  # ImageNet classes
    },
    
    # Evaluation settings
    'eval': {
        'infer_batch_size': 1024,  
        'seed': 42,
        'eval_batch_size': 1024,  
        'num_steps': 10,  # Number of steps for insertion/deletion evaluation
    }
}

# Set environment variables before importing transformers
load_env_config(config['env']['json_path'])

# Now import the rest of the libraries
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
from maskgen.vision_models.vision_maskgen_clip import MaskGeneratingModel
from maskgen.eval.auc_scores import EvalGame

def setup_environment():
    """Setup accelerator"""
    accelerator = Accelerator()
    return accelerator.device

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
    
    # Initialize mask generation model
    exp_base_model = ViTModel.from_pretrained(pretrained_name)
    mask_gen_model = MaskGeneratingModel(
        base_model=exp_base_model,
        hidden_size=model_config.hidden_size,
        num_classes=model_config.num_labels,
        config=None
    )
    mask_gen_model.to(device)
    mask_gen_model.load_state_dict(torch.load(config['env']['model_path']))
    mask_gen_model.eval()
    
    return processor, pred_model, mask_gen_model

def get_transforms(processor):
    """Create image transformation pipeline"""
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    
    if "height" in processor.size:
        size = (processor.size["height"], processor.size["width"])
        crop_size = size
    else:
        size = processor.size["shortest_edge"]
        crop_size = (size, size)
    
    return Compose([
        Resize(size),
        CenterCrop(crop_size),
        ToTensor(),
        normalize,
    ])

def load_data():
    """Load and preprocess the dataset"""
    dataset = load_dataset("mrm8488/ImageNet1K-val")
    dataset = dataset['train']
    splits = dataset.train_test_split(test_size=0.1, seed=config['eval']['seed'])
    test_ds = splits['test']
    splits = splits['train'].train_test_split(test_size=0.1, seed=config['eval']['seed'])
    
    return splits['train'], splits['test'], test_ds

def preprocess_batch(example_batch, transforms):
    """Apply transforms across a batch"""
    example_batch["pixel_values"] = [transforms(image.convert("RGB")) 
                                   for image in example_batch["image"]]
    return example_batch

def collate_fn(examples):
    """Collate function for DataLoader"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def save_results(results, filename):
    """Save evaluation results to disk"""
    results_path = os.path.join(config['env']['results_dir'], filename)
    np.save(results_path, results)


class ViTEvalGame(EvalGame):
    """Extended EvalGame class to handle ViTForImageClassification outputs with memory optimization"""
    
    @torch.no_grad()
    def batched_inference(self, x):
        """Memory-efficient batched inference"""
        outputs = []
        for i in range(0, len(x), self.batch_size):
            batch = x[i:i + self.batch_size]
            output = self.model(batch).logits
            outputs.append(output.cpu())  # Move to CPU immediately
            torch.cuda.empty_cache()  # Clear GPU cache
        return torch.cat(outputs, dim=0).cuda()  # Move back to GPU for final processing

def evaluate_model(pred_model, mask_gen_model, dataloader, device):
    """Run evaluation using insertion and deletion metrics with memory optimization"""
    num_steps = config['eval']['num_steps']
    eval_game = ViTEvalGame(
        model=pred_model,
        num_steps=num_steps,
        output_dim=config['model']['output_dim'],
        batch_size=config['eval']['eval_batch_size']
    )
    
    insertion_scores = []
    deletion_scores = []
    
    for idx, data in tqdm(enumerate(dataloader), desc="Evaluating"):
        # Process smaller chunks of the batch
        pixel_values = data['pixel_values'].to(device)
        chunk_size = config['eval']['eval_batch_size']
        
        for start_idx in range(0, len(pixel_values), chunk_size):
            end_idx = start_idx + chunk_size
            pixel_chunk = pixel_values[start_idx:end_idx]
            
            # Generate attribution masks
            with torch.no_grad():
                pseudo_label = pred_model(pixel_chunk).logits.argmax(-1).view(-1)
                attribution = mask_gen_model.attribute_img(pixel_chunk, pseudo_label)
                
                # Process one operation at a time and clear cache
                ins_auc = eval_game.get_insertion_score(pixel_chunk, attribution)
                torch.cuda.empty_cache()
                
                del_auc = eval_game.get_deletion_score(pixel_chunk, attribution)
                torch.cuda.empty_cache()
                
                insertion_scores.append(ins_auc.cpu())
                deletion_scores.append(del_auc.cpu())
            
            # Print intermediate results
            print(f"\nBatch {idx}, Chunk {start_idx//chunk_size} Results:")
            print(f"Insertion AUC: {ins_auc.mean().item():.4f}")
            print(f"Deletion AUC: {del_auc.mean().item():.4f}")
    
    # Concatenate all scores on CPU
    insertion_scores = torch.cat(insertion_scores).numpy()
    deletion_scores = torch.cat(deletion_scores).numpy()
    
    return {
        'insertion_auc': insertion_scores,
        'deletion_auc': deletion_scores,
    }

def save_results(results, filename):
    """Save evaluation results to disk"""
    os.makedirs(config['env']['results_dir'], exist_ok=True)  # Ensure directory exists
    results_path = os.path.join(config['env']['results_dir'], filename)
    np.save(results_path, results)
    print(f"Saved results to {results_path}")

def main():
    # Enable memory efficient attention if available
    if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Setup
    device = setup_environment()
    processor, pred_model, mask_gen_model = load_models(device)
    transforms = get_transforms(processor)
    
    # Load and prepare data
    train_ds, val_ds, test_ds = load_data()
    test_ds.set_transform(lambda x: preprocess_batch(x, transforms))
    
    test_dataloader = DataLoader(
        test_ds,
        batch_size=config['eval']['infer_batch_size'],
        collate_fn=collate_fn,
        shuffle=False
    )
    
    # Run evaluation
    try:
        results = evaluate_model(pred_model, mask_gen_model, test_dataloader, device)
        
        # Save results
        save_results(results['insertion_auc'], 'insertion_auc_scores.npy')
        save_results(results['deletion_auc'], 'deletion_auc_scores.npy')
        
        # Print summary statistics
        print("\nFinal Evaluation Results:")
        print(f"Average Insertion AUC: {results['insertion_auc'].mean():.4f}")
        print(f"Average Deletion AUC: {results['deletion_auc'].mean():.4f}")

            
    except RuntimeError as e:
        print(f"Error during evaluation: {e}")
        print("Try reducing batch sizes further or freeing up GPU memory")

if __name__ == "__main__":
    main()