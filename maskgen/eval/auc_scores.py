import torch
import torch.nn.functional as F


def idx_to_selector(idx_tensor, selection_size):
    """
    Convert a labels indices tensor of shape [batch_size] 
    to one-hot encoded tensor [batch_size, selection_size]

    """
    return F.one_hot(idx_tensor, num_classes=selection_size).float()

def convert_mask_patch(
    pixel_values: torch.Tensor,
    mask: torch.Tensor,
    h_patch: int,
    w_patch: int,
    fill_strategy: str = 'global_mean',
    noise_std: float = 0.1
) -> torch.Tensor:
    """
    Apply patch mask to image and fill masked regions using various strategies.
    
    Args:
        pixel_values: Image tensor [N, C, H, W]
        mask: Binary mask [N, num_patches]
        h_patch: Height of patch grid
        w_patch: Width of patch grid
        fill_strategy: How to fill masked regions:
            - 'global_mean': Mean across all spatial dimensions
            - 'local_mean': Mean of unmasked regions only
            - 'zero': Fill with zeros
            - 'noise': Fill with Gaussian noise
            - 'channel_mean': Mean per channel
        noise_std: Standard deviation for Gaussian noise (if fill_strategy='noise')
    
    Returns:
        torch.Tensor: Masked image with filled regions
    """
    # Input validation
    if fill_strategy not in ['global_mean', 'local_mean', 'zero', 'noise', 'channel_mean']:
        raise ValueError(f"Unsupported fill strategy: {fill_strategy}")
    
    # Reshape and upsample mask
    reshaped_mask = mask.view(-1, 1, h_patch, w_patch)
    upsampled_mask = F.interpolate(
        reshaped_mask,
        size=pixel_values.shape[-2:],
        mode='nearest'
    )
    
    # Apply mask to input
    masked_values = pixel_values * upsampled_mask
    
    # Calculate fill values based on strategy
    if fill_strategy == 'zero':
        fill_values = torch.zeros_like(pixel_values)
    
    elif fill_strategy == 'global_mean':
        fill_values = pixel_values.mean(dim=(2, 3), keepdim=True)
    
    elif fill_strategy == 'local_mean':
        # Calculate mean of unmasked regions
        sum_values = (pixel_values * upsampled_mask).sum(dim=(2, 3), keepdim=True)
        count = upsampled_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        fill_values = sum_values / count
    
    elif fill_strategy == 'channel_mean':
        # Calculate mean per channel independently
        fill_values = pixel_values.mean(dim=(2, 3), keepdim=True)
    
    elif fill_strategy == 'noise':
        # Generate Gaussian noise with same mean and scaled standard deviation
        means = pixel_values.mean(dim=(2, 3), keepdim=True)
        noise = torch.randn_like(pixel_values) * noise_std
        fill_values = means + noise
    
    # Apply fill values to masked regions
    return masked_values + (1 - upsampled_mask) * fill_values


def obtain_masks_on_topk(attribution, topk, mode='ins', add_noise=False, epsilon=1e-5):
    """
    Create binary masks based on top-k% attribution values.
    
    Args:
        attribution (torch.Tensor): Attribution scores [N, H_a, W_a]
        topk (int): Percentage of top features to select (0-100)
        mode (str): 'ins' for insertion or 'del' for deletion masks
        add_noise (bool): Whether to add random noise to handle ties
        epsilon (float): Noise magnitude if add_noise is True
        
    Returns:
        torch.Tensor: Binary masks of shape [N, H_a, W_a]
    """
    # Flatten spatial dimensions
    N, H, W = attribution.shape
    flat_attribution = attribution.view(N, -1)  # [N, H*W]
    
    # Add optional noise
    if add_noise:
        flat_attribution = flat_attribution + epsilon * torch.randn_like(flat_attribution)
    
    # Calculate number of elements to keep
    k = int(topk * H * W / 100)
    
    # Get threshold value (the kth largest) for each sample in batch
    thresholds = torch.kthvalue(flat_attribution, 
                              flat_attribution.size(1) - k + 1,  
                              dim=1).values.unsqueeze(1)
    
    # Create masks
    masks = (flat_attribution >= thresholds).float()
    
    # Invert masks for deletion mode
    if mode == 'del':
        masks = 1.0 - masks
    elif mode != 'ins':
        raise ValueError("Mode must be either 'ins' or 'del'")
    
    return masks.view(N, H, W)


def obtain_masked_input_on_topk(
    x: torch.Tensor,
    attribution: torch.Tensor,
    topk: int,
    mode: str = 'ins',
    fill_strategy: str = 'global_mean',
    noise_std: float = 0.1
) -> torch.Tensor:
    """
    Apply attribution-based masks to images and fill masked regions.
    
    Args:
        x: Input images [N, C, H, W]
        attribution: Attribution scores [N, H_a, W_a]
        topk: Percentage of top features to keep/remove (0-100)
        mode: 'ins' for insertion or 'del' for deletion
        fill_strategy: Strategy for filling masked regions:
            - 'global_mean': Global mean per channel
            - 'local_mean': Mean of unmasked regions
            - 'zero': Zero filling
            - 'noise': Gaussian noise
            - 'edge': Edge padding
            - 'reflect': Reflection padding
        noise_std: Standard deviation for noise filling (if using 'noise' strategy)
            
    Returns:
        torch.Tensor: Masked images with filled regions [N, C, H, W]
    """
    # Input validation
    if not 0 <= topk <= 100:
        raise ValueError("topk must be between 0 and 100")
    if x.dim() != 4 or attribution.dim() != 3:
        raise ValueError("Invalid input dimensions")
        
    # Get binary mask from attribution scores
    mask = obtain_masks_on_topk(attribution, topk, mode)
    
    # Get patch dimensions from attribution
    h_patch = attribution.shape[-2]
    w_patch = attribution.shape[-1]
    
    # Use convert_mask_patch with specified filling strategy
    return convert_mask_patch(
        pixel_values=x,
        mask=mask,
        h_patch=h_patch,
        w_patch=w_patch,
        fill_strategy=fill_strategy,
        noise_std=noise_std
    )


def obtain_masks_sequence(
    attribution: torch.Tensor,
    num_steps: int = 100,
    add_noise: bool = True,
    epsilon: float = 1e-5
) -> torch.Tensor:
    """
    Create sequence of masks for batch of attributions.
    
    Args:
        attribution: Attribution scores [N, H, W]
        num_steps: Number of masks per attribution
        add_noise: Whether to add noise to handle ties
        epsilon: Noise magnitude if add_noise is True
    
    Returns:
        torch.Tensor: Sequence of binary masks [N, num_steps, H, W]
    """
    N, H, W = attribution.shape
    flat_attr = attribution.reshape(N, -1)  # [N, H*W]
    
    if add_noise:
        flat_attr = flat_attr + epsilon * torch.randn_like(flat_attr)
    
    # Calculate percentile thresholds for each sample in batch
    percentiles = torch.linspace(0, 100, num_steps, device=attribution.device)
    
    # Process each sample separately to avoid memory issues
    masks_list = []
    for attr in flat_attr:
        thresholds = torch.quantile(attr, 1 - percentiles/100)
        masks = (attr.unsqueeze(0) > thresholds.unsqueeze(1)).float()
        masks_list.append(masks)
    
    # Stack masks for all samples [N, num_steps, H*W]
    masks = torch.stack(masks_list)
    
    return masks.view(N, num_steps, H, W)
    

def obtain_masked_inputs_sequence(
    x: torch.Tensor,
    attribution: torch.Tensor,
    num_steps: int,
    mode: str = 'ins',
    fill_strategy: str = 'global_mean'
) -> torch.Tensor:
    """
    Create sequence of masked inputs for batch of images.
    
    Args:
        x: Input images [N, C, H, W]
        attribution: Attribution scores [N, H_a, W_a]
        mode: 'ins' for insertion or 'del' for deletion
        fill_strategy: Strategy for filling masked regions
        
    Returns:
        torch.Tensor: Sequence of masked inputs [N*num_steps, C, H, W]
    """
    N = x.size(0)
    
    # Get sequence of masks [N, num_steps, H, W]
    masks_sequence = obtain_masks_sequence(attribution, num_steps=num_steps)  
    
    # Reshape masks to [N*num_steps, 1, H, W]
    masks_sequence = masks_sequence.reshape(-1, 1, *masks_sequence.shape[-2:])

    # Invert masks for deletion
    if mode == 'del':
        masks_sequence = 1.0 - masks_sequence
    elif mode != 'ins':
        raise ValueError("Mode must be either 'ins' or 'del'")
    
    # Expand input to match number of steps
    x_expanded = x.repeat_interleave(num_steps, dim=0)  # [N*num_steps, C, H, W]
    
    # Apply masks and fill masked regions
    return convert_mask_patch(
        pixel_values=x_expanded,
        mask=masks_sequence.squeeze(1),  # Remove channel dim for convert_mask_patch
        h_patch=attribution.shape[-2],
        w_patch=attribution.shape[-1],
        fill_strategy=fill_strategy
    ) # [N*num_steps, C, H, W]


class EvalGame:
    """
    Evaluation games for attribution methods with batch processing support.
    """
    def __init__(self, model, num_steps=10, output_dim=1000, batch_size=512):
        """
        Args:
            model: Model that takes input and outputs logits
            output_dim: Output dimension of the model
            batch_size: Batch size for inference to manage memory
        """
        self.model = model
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.num_steps = num_steps
        
    def get_insertion_score(self, x, attribution):
        """Get insertion AUC scores for batch."""
        return self.get_auc(x, attribution, 'ins')
        
    def get_deletion_score(self, x, attribution):
        """Get deletion AUC scores for batch."""
        return self.get_auc(x, attribution, 'del')
    
    @torch.no_grad()  # Add no_grad for efficiency
    def batched_inference(self, x):
        """
        Run model inference in batches to manage memory.
        
        Args:
            x: Input tensor [N, C, H, W] or [N*num_steps, C, H, W]
            
        Returns:
            torch.Tensor: Model outputs
        """
        outputs = []
        for i in range(0, len(x), self.batch_size):
            # Python's slicing automatically handles the last incomplete batch 
            # - if the end index is beyond the array length, 
            # it just takes whatever is available.
            batch = x[i:i + self.batch_size] 
            output = self.model(batch)
            outputs.append(output)
        return torch.cat(outputs, dim=0)
    
    def play_game(self, x, attribution, num_steps, mode='ins'):
        """
        Play insertion/deletion game with batch support.
        
        Args:
            x: Input images [N, C, H, W]
            attribution: Attribution scores [N, H_a, W_a]
            mode: 'ins' for insertion or 'del' for deletion
            
        Returns:
            torch.Tensor: Probabilities for each step [N, num_steps]
        """
        N = x.size(0)
        
        # Get pseudo-labels for full batch
        pseudo_labels = self.batched_inference(x).argmax(-1)  # [N]
        selector = idx_to_selector(pseudo_labels, self.output_dim)  # [N, output_dim]
        
        # Get masked sequence for full batch
        x_sequence = obtain_masked_inputs_sequence(x, attribution, num_steps=num_steps, mode=mode)  # [N*100, C, H, W]
        
        # Run inference in batches
        logits = self.batched_inference(x_sequence)  # [N*100, output_dim]
        probs = torch.softmax(logits, dim=-1)
        
        # Reshape to separate batch and steps dimensions
        probs = probs.view(N, -1, self.output_dim)  # [N, 100, output_dim]
        
        # Apply selector for each sample in batch
        selector = selector.unsqueeze(1)  # [N, 1, output_dim]
        probs = (probs * selector).sum(-1)  # [N, 100]
        
        return probs # [N, 100]
    
    def get_auc(self, x, attribution, mode='ins'):
        """
        Calculate AUC scores for batch.
        
        Returns:
            torch.Tensor: AUC scores [N]
        """
        num_steps = self.num_steps
        probs = self.play_game(x, attribution, num_steps, mode)  # [N, num_steps]
        x_axis = torch.linspace(0, 1, probs.size(1), device=probs.device)
        return torch.trapezoid(probs, x_axis, dim=1)  # [N]
    
    def get_score_at_topk(self, x, attribution, topk, mode='ins'):
        """
        Get scores at specific topk percentage for batch.
        
        Args:
            x: Input images [N, C, H, W]
            attribution: Attribution scores [N, H_a, W_a]
            topk: Integer [0, 100]
            mode: 'ins' for insertion or 'del' for deletion
            
        Returns:
            torch.Tensor: Scores [N]
        """
        # Get masked inputs for full batch
        masked_input = obtain_masked_input_on_topk(x, attribution, topk, mode)
        
        # Get pseudo-labels for full batch
        pseudo_labels = self.batched_inference(x).argmax(-1)  # [N]
        selector = idx_to_selector(pseudo_labels, self.output_dim)  # [N, output_dim]
        
        # Get probabilities for masked inputs
        logits = self.batched_inference(masked_input)  # [N, output_dim]
        probs = torch.softmax(logits, dim=-1)
        
        # Apply selector and get final scores
        return (probs * selector).sum(-1)  # [N]
    
    def get_insertion_at_topk(self, x, attribution, topk):
        """Get insertion scores at topk for batch."""
        return self.get_score_at_topk(x, attribution, topk, mode='ins')
    
    def get_deletion_at_topk(self, x, attribution, topk):
        """Get deletion scores at topk for batch."""
        return self.get_score_at_topk(x, attribution, topk, mode='del')

