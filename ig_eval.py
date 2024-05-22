import torch 
import torch.nn as nn
import torch.nn.functional as F
import os
import transformers
from accelerate import Accelerator
from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel, ViTConfig, TrainingArguments, Trainer
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import cv2
from datasets import load_dataset,load_metric
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from captum.attr import IntegratedGradients
import sys

accelerator = Accelerator()
device = accelerator.device



pretrained_name = 'google/vit-base-patch16-224'
# pretrained_name = 'vit-base-patch16-224-finetuned-imageneteval'
# pretrained_name = 'openai/clip-vit-base-patch32'
config = ViTConfig.from_pretrained(pretrained_name)
processor = ViTImageProcessor.from_pretrained(pretrained_name)
# get mean and std to unnormalize the processed images
mean, std = processor.image_mean, processor.image_std

pred_model = ViTForImageClassification.from_pretrained(pretrained_name)
pred_model.to(device)
# set to eval mode
pred_model.eval()


# def get_heatmap(attribution_ig):
#     # heatmap = torch.relu(attribution_ig.sum(dim=1))
#     heatmap = attribution_ig.sum(dim=1)
#     # Average pooling to convert to 14*14 heatmap
#     heatmap = F.avg_pool2d(heatmap, kernel_size=16, stride=16)
#     heatmap = heatmap.squeeze(0).detach().cpu().numpy()
#     return heatmap


# heatmap = get_heatmap(attribution_ig)


from torch.utils.data import DataLoader
def load_data(seed=42): 
    dataset = load_dataset("mrm8488/ImageNet1K-val")
    dataset = dataset['train']
    splits = dataset.train_test_split(test_size=0.1, seed=seed)
    test_ds = splits['test']
    splits = splits['train'].train_test_split(test_size=0.1, seed=seed)
    train_ds = splits['train']
    val_ds = splits['test']
    return train_ds, val_ds, test_ds

_, _, test_ds = load_data()

normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
if "height" in processor.size:
    size = (processor.size["height"], processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in processor.size:
    size = processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = processor.size.get("longest_edge")

transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

test_ds.set_transform(preprocess)

# batch size is limited to 2, because n_steps could could huge memory consumption
batch_size = 1
test_dataloader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)



root_path = 'results/ig-vit'

if not os.path.exists(root_path):
    os.makedirs(root_path)
    print("Folder created successfully.")
else:
    print("Folder already exists.")


model = lambda x: pred_model(pixel_values=x).logits

ig = IntegratedGradients(lambda x: torch.softmax(model(x), dim=-1))

def save_heatmap(heatmap, batch_size, idx, root_path):
    # convert to numpy
    heatmap = heatmap.detach().cpu().numpy() # [N, 224, 224]
    tail_index = batch_size * (idx + 1)
    file_name = os.path.join(root_path, f"heatmap-{tail_index}")
    np.save(file_name, heatmap)


ins_score_list = []
del_score_list = []
heatmap_list = []


from tqdm import tqdm

for idx, data in tqdm(enumerate(test_dataloader)):
    pixel_values = data['pixel_values'].to(device)
    with torch.no_grad():
        pseudo_label = model(pixel_values).argmax(-1).view(-1)

    # sum up all 3 RGB channels for heatmap
    attribution_ig = ig.attribute(pixel_values, target=pseudo_label, n_steps=100)
    heatmap = attribution_ig.sum(dim=1)
    heatmap_list.append(heatmap)

    with torch.no_grad():
        # Average pooling to convert to 14*14 heatmap
        heatmap = F.avg_pool2d(heatmap, kernel_size=16, stride=16)
    
    # if idx == 10:
    #     break

if len(heatmap_list) > 0:
    heatmap_cat_tensor = torch.cat(heatmap_list, dim=0) # (N, 224, 224)
    save_heatmap(heatmap_cat_tensor, batch_size, idx, root_path)

print("Integrated Gradients completed successfully.")