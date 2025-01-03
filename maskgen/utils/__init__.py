from maskgen.utils.data_utils import load_imagenet
from maskgen.utils.model_utils import get_pred_model
from maskgen.utils.image_utils import get_preprocess, collate_fn
from accelerate import Accelerator
from torch.nn import functional as F
from maskgen.utils.img_utils import plot_overlap_np

__all__ = ['load_imagenet', 'get_pred_model', 'get_device', 'idx_to_selector'
           , 'get_preprocess', 'collate_fn', 'plot_overlap_np']

def get_device():
    accelerator = Accelerator()
    return accelerator.device

def idx_to_selector(idx_tensor, selection_size):
    """
    Convert a labels indices tensor of shape [batch_size] 
    to one-hot encoded tensor [batch_size, selection_size]

    """
    return F.one_hot(idx_tensor, num_classes=selection_size).float()