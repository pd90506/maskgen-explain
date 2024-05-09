import torch 
from utils.img_utils import generate_mask
from utils import idx_to_selector
import torch.nn.functional as F


class RandomMaskSaliency():
    def __init__(self, model, num_classes):
        """
        Initialize the RandomMask class.

        Args:
            model: The model that outputs logits.
            num_classes: The number of classes.

        Returns:
            None
        """
        super().__init__()

        self.model = model
        self.num_classes = num_classes
    

    def attribute_img(self, 
                      x, 
                      image_size=224, 
                      patch_size=16, 
                      baseline=None, 
                      n_samples=1000,
                      mask_prob=0.7,
                      seed=None):
        """
        Generate attribution heatmap for an input image.

        Args:
            x: An image tensor of shape [1, C, H, W], where H = W = image_size.
            image_size: The size of the input image (H = W = image_size).
            patch_size: The size of each patch. Can be used to calculate the number of tokens in each patch 
                        (image_size // patch_size) ** 2.
            baseline: The baseline tensor. If None, the baseline is set to the zero tensor.
            n_samples: The number of random masks to be generated.
            mask_prob: The probability of a token being masked, i.e., replaced by the baseline.
            seed: The seed value for random number generation.

        Returns:
            Attribution heatmap.
        """

        device = x.device
        size = image_size // patch_size
        with torch.no_grad():
            # Get the original prediction idx and the selector
            predicted_class_idx = self.model(x).argmax(-1) # [1,]
            selector = idx_to_selector(predicted_class_idx, self.num_classes) # [1, n_classes]

            # Generate a random mask and reshape to the proper size.
            mask = generate_mask(mask_size=size*size, mask_probability=mask_prob, batch_size=n_samples, seed=seed)
            mask = mask.to(device)
            mask = mask.reshape(-1, 1, size, size) # [N, 1, size, size]

            # Interpolate mask to the size of the original image.
            masked_pixels = F.interpolate(mask, x.shape[-2:], mode='nearest') # [N, 1, H, W]

            # A number of `n_samples` randomly masked inputs
            x_masked = x * (1 - masked_pixels) # [N, C, H, W]

            # Obtain the output probabilities of the masked inputs for the true predicted class.
            logits = self.model(x_masked) # [N, n_classes]
            probs = torch.softmax(logits, dim=-1) # [N, n_classes]

            # Only consider the masks with correct predictions. Weighted usign probs
            correct_idx = (logits.argmax(-1) == predicted_class_idx).float().unsqueeze(-1) # [N, 1]
            probs = ((probs * selector) * correct_idx).sum(-1, keepdim=True) # [N, 1]
            probs = probs.unsqueeze(-1).unsqueeze(-1) # [N, 1, 1, 1]

            # Weighted mask
            weighted_mask = (probs * (1 - mask)).sum(0, keepdim=True) / probs.sum() # [1, 1, size, size]
        
        return weighted_mask
    
    def attribute_text(self, x, baseline=None, n_samples=1000,):
        # TODO
        raise NotImplementedError("This function hasn't been developed.")


