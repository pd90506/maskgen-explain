from maskgen.utils.model_utils import load_exp_and_target_model
from maskgen.utils.data_utils import get_imagenet_dataloader
import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os


class PixelHeatmapDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.pixel_values = data['pixel_values']
        self.heatmaps = data['heatmaps']
        
    def __len__(self):
        return len(self.pixel_values)
    
    def __getitem__(self, idx):
        return {
            'pixel_values': torch.from_numpy(self.pixel_values[idx]),
            'heatmap': torch.from_numpy(self.heatmaps[idx])
        }


def save_pixel_heatmap_pairs(pixel_values, heatmaps, save_path):
    """Save pixel values and heatmaps as paired data"""
    # Convert to numpy if needed
    if torch.is_tensor(pixel_values):
        pixel_values = pixel_values.cpu().numpy()
    if torch.is_tensor(heatmaps):
        heatmaps = heatmaps.cpu().numpy()
    
    # Save paired data
    np.savez(save_path, 
             pixel_values=pixel_values, 
             heatmaps=heatmaps)


def load_pixel_heatmap_pairs(npz_path, batch_size=32, shuffle=True):
    """Load saved pairs into a DataLoader"""
    dataset = PixelHeatmapDataset(npz_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def save_maskgen_results(config, device):
    all_inputs = []
    all_heatmaps = []
    save_path = config['results_path']

    # get models 
    target_model, maskgen_model, processor = load_exp_and_target_model(config, device)

    # get dataloader
    dataloader = get_imagenet_dataloader(split='tiny', 
                                         batch_size=config['batch_size'], 
                                         processor=processor, 
                                         shuffle=False,
                                         num_samples=config['num_samples'])

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing batches"):
        inputs = batch['pixel_values'].to(device)
        with torch.no_grad():
            predicted_class_idx = target_model(inputs).logits.argmax(-1)
            heatmap = maskgen_model.attribute_img(inputs, predicted_class_idx.unsqueeze(1), image_size=224, patch_size=16)
            
            # Convert current batch to numpy and append to lists
            inputs_np = inputs.cpu().numpy()
            heatmap_np = heatmap.cpu().numpy()
            all_inputs.append(inputs_np)
            all_heatmaps.append(heatmap_np)
            
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_heatmaps = np.concatenate(all_heatmaps, axis=0)
    # ensure save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, 'pixel_heatmap_pairs.npz')
    print("Saving pixel-heatmap pairs to:", save_path)
    save_pixel_heatmap_pairs(all_inputs, all_heatmaps, save_path)

