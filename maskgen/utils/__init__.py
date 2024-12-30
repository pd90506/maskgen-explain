from .data_utils import load_imagenet_eval_data
from .model_utils import get_pred_model

from accelerate import Accelerator
from torch.nn import functional as F

__all__ = ['load_imagenet_eval_data', 'get_pred_model', 'get_device', 'idx_to_selector']

def get_device():
    accelerator = Accelerator()
    return accelerator.device

def idx_to_selector(idx_tensor, selection_size):
    """
    Convert a labels indices tensor of shape [batch_size] 
    to one-hot encoded tensor [batch_size, selection_size]

    """
    return F.one_hot(idx_tensor, num_classes=selection_size).float()