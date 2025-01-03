import torch
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize
from transformers import ViTImageProcessor
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


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

def normalize_and_rescale(heatmap: np.ndarray) -> np.ndarray:
    """Normalize heatmap to [0, 255] range."""
    max_value = np.max(heatmap)
    min_value = np.min(heatmap)
    heatmap_ft = (heatmap - min_value) / (max_value - min_value)
    return (heatmap_ft * 255).astype(np.uint8)

def save_heatmap(heatmap: np.ndarray, save_path: str, image_id: int):
    """Save a 14x14 heatmap visualization.
    
    Args:
        heatmap: numpy array of shape (14, 14)
        save_path: directory to save the heatmap
        image_id: identifier for the image
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap='viridis')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'heatmap_{image_id}.png'))
    plt.close()

def unnormalize(img: np.ndarray, mean: list, std: list) -> np.ndarray:
    """Unnormalize image from model input space."""
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    return img * std + mean

def prepare_overlap(image: np.ndarray, heatmap: np.ndarray, 
                   img_mean: list, img_std: list) -> tuple:
    """Prepare image and heatmap for overlap visualization."""
    # Process heatmap
    heatmap = normalize_and_rescale(heatmap)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[2]))
    blur = cv2.GaussianBlur(heatmap, (13, 13), 11)
    heatmap_colored = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Process image
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image = unnormalize(image, img_mean, img_std)
    image = (image * 255).astype(np.uint8)
    
    # Create overlap
    overlap = cv2.addWeighted(heatmap_colored, 0.5, image, 0.5, 0)
    
    return image, overlap

def plot_overlap(image: np.ndarray, heatmap: np.ndarray, 
                img_mean: list, img_std: list, 
                save_path: str, image_id: int, 
                both: bool = False):
    """Plot and save overlap visualization.
    
    Args:
        image: numpy array of shape (C, H, W)
        heatmap: numpy array of shape (14, 14)
        img_mean: normalization mean
        img_std: normalization std
        save_path: directory to save visualization
        image_id: identifier for the image
        both: if True, plot original image alongside overlap
    """
    original_img, overlap = prepare_overlap(image, heatmap, img_mean, img_std)
    
    if both:
        plt.figure(figsize=(12, 6))
        
        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.axis('off')
        plt.title('Original Image')
        
        # Plot overlap
        plt.subplot(1, 2, 2)
        plt.imshow(overlap)
        plt.axis('off')
        plt.title('Attention Overlap')
        
    else:
        plt.figure(figsize=(6, 6))
        plt.imshow(overlap)
        plt.axis('off')
        plt.title('Attention Overlap')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'overlap_{image_id}.png'))
    plt.close()