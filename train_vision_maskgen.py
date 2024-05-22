import torch 
from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel, ViTConfig
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

from accelerate import Accelerator

import logging
import sys

accelerator = Accelerator()
device = accelerator.device


# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# url = "http://farm3.staticflickr.com/2066/1798910782_5536af8767_z.jpg"
# url = "http://farm1.staticflickr.com/184/399924547_98e6cef97a_z.jpg"
url = "http://farm1.staticflickr.com/128/318959350_1a39aae18c_z.jpg"
image = Image.open(requests.get(url, stream=True).raw)

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

with torch.no_grad():
    inputs = processor(images=image, return_tensors="pt")
    inputs.to(device)
    outputs = pred_model(**inputs, output_hidden_states=True)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", pred_model.config.id2label[predicted_class_idx])



from maskgen.models.vision_maskgen_model9 import MaskGeneratingModel

mask_gen_model = MaskGeneratingModel(pred_model, hidden_size=config.hidden_size, num_classes=config.num_labels)
mask_gen_model.to(device)
print()


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

train_ds, val_ds, test_ds = load_data()
dataset = load_dataset("mrm8488/ImageNet1K-val")['train']

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
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
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

train_ds.set_transform(preprocess)
test_ds.set_transform(preprocess)
dataset.set_transform(preprocess)


batch_size = 256
# train_dataloader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers)
# train_dataloader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers)
train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)


n_steps = 5
n_samples = 5

params_to_optimize = [name for name, param in mask_gen_model.named_parameters() if param.requires_grad]
print("params to be optimized: ")
print(params_to_optimize)


from tqdm import tqdm

params_to_optimize = [param for param in mask_gen_model.parameters() if param.requires_grad]
# optimizer = torch.optim.Adam(params_to_optimize, lr=1e-3, weight_decay=1e-5)
optimizer = torch.optim.Adam(params_to_optimize, lr=1e-3, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)


logging.basicConfig(level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('mask_gen_model/app.log', 'w', 'utf-8'),
                    logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

for epoch in range(10):
    pbar = tqdm(train_dataloader)
    for idx, data in enumerate(pbar):
        pixel_values = data['pixel_values'].to(device)
        loss_dict = mask_gen_model.train_one_batch(pixel_values, optimizer=optimizer, n_steps=n_steps, n_samples=n_samples)
        # scheduler.step()
        pbar.set_description(f"Epoch {epoch+1}, Step {idx+1}: Loss = {loss_dict['loss'].item():.4f}, " 
                             f"Reward Loss = {loss_dict['reward_loss'].item():.4f}, "
                            #  f"Regret Loss = {loss_dict['regret_loss'].item():.4f}, "
                             f"Mask Loss = {loss_dict['mask_loss'].item():.4f} "
                            #  f"alt_mask_loss = {loss_dict['alt_mask_loss'].item():.4f} "
                             f"mask_mean = {loss_dict['mask_mean'].item():.4f} "
                             f"prob_mean = {loss_dict['prob_mean'].item():.4f} "
                             )
        if idx % 10 == 0:
            # Use logger instead of print
            logger.info(f"Epoch {epoch+1}, Step {idx+1}: Loss = {loss_dict['loss'].item():.4f}, " 
                             f"Reward Loss = {loss_dict['reward_loss'].item():.4f}, "
                            #  f"Regret Loss = {loss_dict['regret_loss'].item():.4f}, "
                             f"Mask Loss = {loss_dict['mask_loss'].item():.4f} "
                            #  f"alt_mask_loss = {loss_dict['alt_mask_loss'].item():.4f} "
                             f"mask_mean = {loss_dict['mask_mean'].item():.4f} "
                             f"prob_mean = {loss_dict['prob_mean'].item():.4f} "
                             )
        if (idx) % 10 == 0:
            # Configure logging

            
            torch.save(mask_gen_model.state_dict(), f'mask_gen_model/mask_gen_model_{epoch}_{idx}.pth') 



torch.save(mask_gen_model.state_dict(), f'mask_gen_model/mask_gen_model_final_{epoch}_{idx}.pth') 
