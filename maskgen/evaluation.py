import torch
import torch.nn.functional as F

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
    reshaped_mask = mask.reshape(-1, h_patch, w_patch).unsqueeze(1) # [N, 1, h_patch ,w_patch]
    image_size = pixel_values.shape[-2:]
    reshaped_mask = torch.nn.functional.interpolate(reshaped_mask, size=image_size, mode='nearest')
    return pixel_values * reshaped_mask + (1 - reshaped_mask) * pixel_values.mean(dim=(-1,-2), keepdim=True)


def obtain_masks_on_topk(attribution, topk, mode='ins'):
    """ 
    attribution: [N, H_a, W_a]
    """
    H_a, W_a = attribution.shape[-2:]
    attribution = attribution.reshape(-1, H_a * W_a) # [N, H_a*W_a]
    attribution_perturb = attribution + 1e-5*torch.randn_like(attribution) # to avoid equal attributions (typically all zeros or all ones)
    
    attribution_size = H_a * W_a
    a, _ = torch.topk(attribution_perturb, k=int(topk * attribution_size / 100), dim=-1)
    a = a[:, -1].unsqueeze(-1)
    mask = (attribution_perturb >= a).float()
    if mode == 'ins':
        pass 
    elif mode == 'del':
        mask = 1. - mask
    else:
        raise ValueError('Enter game mode either as ins or del.')
    return mask.reshape(-1, H_a, W_a) # [N, H_a*W_a]


def obtain_masked_input_on_topk(x, attribution, topk, mode='ins'):
    """ 
    x: [N, C, H, W]
    attribution: [N, H_a, W_a]
    """
    mask = obtain_masks_on_topk(attribution, topk, mode)
    mask = mask.unsqueeze(1) # [N, 1, H_a, W_a]
    mask = F.interpolate(mask, size=x.shape[-2:], mode='nearest')

    masked_input = x * mask
    # mean_pixel = masked_input.sum(dim=(-1, -2), keepdim=True) / mask.sum(dim=(-1, -2), keepdim=True)
    # mean_pixel = x.mean(dim=(-1, -2), keepdim=True)
    # masked_input = masked_input + (1 - mask) * mean_pixel
    return masked_input


def obtain_masks_sequence(attribution):
    """ 
    attribution: [1, H_a, W_a]
    """
    H_a, W_a = attribution.shape[-2:]
    attribution_size = H_a * W_a
    attribution = attribution.reshape(-1, H_a * W_a) # [1, H_a*W_a]
    attribution = attribution + 1e-4 * torch.randn_like(attribution)
    a, _ = torch.sort(attribution, dim=-1, descending=True)
    idx = torch.ceil(torch.arange(100) * attribution_size / 100).int()
    a = a.reshape(-1, 1)
    a = a[idx,:] # [100, 1]
    res = (attribution > a).float() # [100, H_a*W_a]
    return res.reshape(-1, H_a, W_a)
    

def obtain_masked_inputs_sequence(x, attribution, mode='ins'):
    """ 
    x: [1, C, H, W]
    attribution: [1, H_a, W_a]
    """
    masks_sequence = obtain_masks_sequence(attribution) # [100, H_a, W_a]
    masks_sequence = masks_sequence.unsqueeze(1) # [100, 1, H_a, W_a]
    masks_sequence = F.interpolate(masks_sequence, size=x.shape[-2:], mode='nearest') # [100, 1, H, W]
    if mode == 'del':
        masks_sequence = 1.0 - masks_sequence
    elif mode == 'ins':
        pass
    else:
        raise ValueError('Enter game mode either as ins or del.')
    return x * masks_sequence # [100, C, H, W]


class EvalGame():
    """ Evaluation games
    """
    def __init__(self, model, output_dim=1000):
        """ 
        model: a prediction model takes an input and outputs logits
        """
        self.model = model
        self.output_dim = output_dim
    
    def get_insertion_score(self, x, attribution):
        return self.get_auc(x, attribution, 'ins')
    
    def get_deletion_score(self, x, attribution):
        return self.get_auc(x, attribution, 'del')
    
    def play_game(self, x, attribution, mode='ins'):
        """ 
        masking the input with a series of masks based on the attribution importance.
        x: [1, C, H, W] the batch dim must be 1
        attribution: [1, H_a, W_a] the batch dim must be 1

        """
        pseudo_label = self.model(x).argmax(-1) # [1, 1]
        selector = idx_to_selector(pseudo_label, self.output_dim) # [1, 1000]
        
        x_sequence = obtain_masked_inputs_sequence(x, attribution, mode=mode) # [100, C, H, W]
        probs = torch.softmax(self.model(x_sequence), dim=-1) # [100, 1000]
        probs = (probs * selector).sum(-1) # [100,]
        
        return probs
    
    def get_auc(self, x, attribution, mode='ins'):
        probs = self.play_game(x, attribution, mode)
        return probs.sum()
    
    def get_insertion_at_topk(self, x, attribution, topk):
        """"
        obtain insertion score at top k, i.e, probability of predicted class
        when inserting top k patches
        """
        return self.get_score_at_topk(x, attribution, topk, mode='ins')
    
    def get_deletion_at_topk(self, x, attribution, topk):
        """"
        obtain deletion score at top k, i.e, probability of predicted class
        when deleting top k patches
        """
        return self.get_score_at_topk(x, attribution, topk, mode='del') 
    
    def get_score_at_topk(self, x, attribution, topk, mode='ins'):
        """ 
        x: [N, C, H, W]
        attribution: [N, H_a, W_a]
        topk: Integer [0, 100]
        """
        masked_input = obtain_masked_input_on_topk(x, attribution, topk, mode)

        pseudo_label = self.model(x).argmax(-1) # [1, 1]
        probs = torch.softmax(self.model(masked_input), dim=-1) # [N, 1000]

        selector = idx_to_selector(pseudo_label, self.output_dim) # [N, 1000]
        probs = (probs * selector).sum(-1) # [N,]
        return probs # [N,]

