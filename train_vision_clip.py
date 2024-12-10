import os
import json 
import logging
from datetime import datetime
from tqdm import tqdm

from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel, ViTConfig
from datasets import load_dataset,load_metric
from maskgen.vision_models.vision_maskgen_clip import MaskGeneratingModel

# ====================== Global parameters starts ======================
logging.basicConfig(
    filename='log/app.log',            # Specify the log file name
    level=logging.DEBUG,           # Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Set the log format
)

# Load the environment configuration JSON data
json_path = 'env_config.json'
with open(json_path, 'r') as file:
    env_config = json.load(file)

hf_home = env_config['HF_HOME']
# Set the HF_HOME environment variable
os.environ['HF_HOME'] = hf_home
# Set the access token to huggingface hub
access_token = env_config['access_token']
os.environ['HUGGINGFACE_HUB_TOKEN'] = access_token
num_workers = 8
# ====================== Global parameters ends ======================


def load_data(seed=42): 
    dataset = load_dataset("mrm8488/ImageNet1K-val")
    dataset = dataset['train']
    splits = dataset.train_test_split(test_size=0.1, seed=seed)
    test_ds = splits['test']
    splits = splits['train'].train_test_split(test_size=0.1, seed=seed)
    train_ds = splits['train']
    val_ds = splits['test']
    return train_ds, val_ds, test_ds

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def get_dataloader(dataset, processor, batch_size):
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
    dataset.set_transform(preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers)
    return dataloader


def save_model(model, model_prefix):
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    model_name = model_prefix + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pt"
    model_path = os.path.join("trained", model_name)
    torch.save(state_dict, model_path)


def main():
    accelerator = Accelerator()
    device = accelerator.device
    print(torch.cuda.device_count())
    batch_size = 512
    num_steps = 5
    mini_batch_size = 256
    ppo_epochs = 1

    # get data and dataloaders
    train_ds, val_ds, test_ds = load_data()
    train_dataloader = get_dataloader(train_ds, processor, batch_size=batch_size)

    # define the prediction model
    pretrained_name = 'google/vit-base-patch16-224'
    config = ViTConfig.from_pretrained(pretrained_name)
    processor = ViTImageProcessor.from_pretrained(pretrained_name)
    # mean, std = processor.image_mean, processor.image_std
    pred_model = ViTForImageClassification.from_pretrained(pretrained_name)
    pred_model.to(device)
    # set to evaluation mode
    pred_model.eval()

    # Create the explainer g(x)
    exp_base_model = ViTModel.from_pretrained(pretrained_name) 
    mask_gen_model = MaskGeneratingModel(base_model=exp_base_model, hidden_size=config.hidden_size, num_classes=config.num_labels)
    mask_gen_model.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(mask_gen_model.parameters(), lr=1e-4)

    for epoch in range(5):
        pbar = tqdm(train_dataloader)
        for idx, data in enumerate(pbar):
            pixel_values = data['pixel_values'].to(device)

            loss_dict = mask_gen_model.train_one_batch(pred_model, pixel_values, optimizer, num_steps=num_steps, mini_batch_size=mini_batch_size, ppo_epochs=ppo_epochs)
            desc = f"Epoch {epoch+1}, Step {idx+1}: Loss = {loss_dict['loss']:.4f}, " \
                f"Actor Loss = {loss_dict['actor_loss']:.4f}, " \
                f"Critic Loss = {loss_dict['critic_loss']:.4f}, " \
                f"Entropy = {loss_dict['entropy']:.4f}, " \
                f"Returns = {loss_dict['returns']:.4f}, " \
                f"Value = {loss_dict['value']:.4f}, " \
                f"kl_div = {loss_dict['kl_div']:.4f}, " \

            pbar.set_description(desc)
            if (idx + 1) % 50 == 0:
                print(f"{desc}")
                # save the model
                save_model(mask_gen_model, "vision_clip_")



if __name__ == "__main__":
    main()