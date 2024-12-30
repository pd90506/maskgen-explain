import torch
import torch.nn.functional as F
import numpy as np


def idx_to_selector(idx_tensor, selection_size):
    """
    Convert a labels indices tensor of shape [batch_size] 
    to one-hot encoded tensor [batch_size, selection_size]

    """
    batch_size = idx_tensor.shape[0]
    dummy = torch.arange(selection_size, device=idx_tensor.device).unsqueeze(0).expand(batch_size, -1)
    extended_idx_tensor = idx_tensor.unsqueeze(-1).expand(-1, selection_size)
    return (dummy == extended_idx_tensor).float()

def convert_mask_patch(pixel_values, mask, h_patch, w_patch):
    """
    given pixel values and mask, return masked pixel values
    """
    reshaped_mask = mask.reshape(-1, h_patch, w_patch).unsqueeze(1) # [N, 1, h_patch ,w_patch]
    image_size = pixel_values.shape[-2:]
    reshaped_mask = torch.nn.functional.interpolate(reshaped_mask, size=image_size, mode='nearest')
    return pixel_values * reshaped_mask + (1 - reshaped_mask) * pixel_values.mean(dim=(-1,-2), keepdim=True)


def batch_combine(x, func, n_batch_dims=1):
    """ 
    Temperarily combine multiple dimensions into one batch dimension in order to feed to functions
    that doesn't support multiple batch dimensions.
    Args:
        x: input
        func: function to be passed to
        batch_dims: number of dimensions to be combined, starting from the 0th dimension
    return:
        output having the corresponding shape of x
    """
    shape = x.shape
    batch_shape, rest_shape = shape[:n_batch_dims + 1], shape[n_batch_dims + 1:]
    combined_batch_shape = [np.prod(batch_shape),]
    res = func(x.reshape(combined_batch_shape + rest_shape))
    return res.reshape(batch_shape + rest_shape)