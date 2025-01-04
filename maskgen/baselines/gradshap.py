import torch
from captum.attr import GradientShap
import numpy as np
from torch import nn
import requests
from PIL import Image

class ModelWrapper(nn.Module):
    """
    Wrapper class for models that return dictionaries containing logits.
    Adapts the forward pass to be compatible with Captum's attribution methods.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        """
        Modify forward pass to return only the logits tensor instead of a dictionary.
        """
        outputs = self.model(pixel_values=x)
        return outputs.logits

def prepare_model(model):
    """
    Prepare a model for use with GradShap attribution.
    
    Args:
        model: Original model that returns dictionary with logits
        
    Returns:
        ModelWrapper: Wrapped model compatible with Captum
    """
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()  # Set to evaluation mode
    return wrapped_model

class GradShapAnalyzer:
    """
    A class to compute GradientShap attribution for a single image.
    """
    
    def __init__(
        self,
        model,
        device,
        n_samples=50,
        baseline_std=0.001,
        stdevs=0.09
    ):
        """
        Initialize the GradShapAnalyzer.
        
        Args:
            model: The prediction model
            processor: Image processor/transform
            device: Computing device (CPU/GPU)
            n_samples: Number of samples for GradientShap
            baseline_std: Standard deviation for baseline noise
            stdevs: Standard deviation for GradientShap
        """
        # Prepare model for attribution
        self.model = prepare_model(model).to(device)
        self.device = device
        self.n_samples = n_samples
        self.baseline_std = baseline_std
        self.stdevs = stdevs
        
        # Initialize GradientShap with wrapped model
        self.gradient_shap = GradientShap(self.model)
    
    def _prepare_baseline(self, input_shape):
        """Create baseline input for GradientShap."""
        return torch.randn(1, *input_shape) * self.baseline_std
    
    def get_attribution(self, pixel_values, target_class=None):
        """
        Compute GradientShap attribution scores for a single image.
        
        Args:
            image: Input image
            target_class: Optional target class index. If None, uses model's prediction
        
        Returns:
            tuple: (attribution_map, predicted_class, prediction_confidence)
        """
        # Process image
        pixel_values = pixel_values.to(self.device)
        
        # Get prediction if target class not provided
        if target_class is None:
            with torch.no_grad():
                output = self.model(pixel_values)
                predicted_class = output.argmax(dim=1).item()
                confidence = torch.softmax(output, dim=1).max().item()
                target_class = predicted_class
        else:
            predicted_class = target_class
            confidence = None
        
        # Create baseline
        baseline = self._prepare_baseline(pixel_values.shape[1:]).to(self.device)
        
        # Compute attribution
        attribution = self.gradient_shap.attribute(
            pixel_values,
            baselines=baseline,
            target=target_class,
            n_samples=self.n_samples,
            stdevs=self.stdevs
        )
        
        # Convert to numpy array
        attribution_map = attribution.squeeze().cpu().detach().numpy()
        
        return attribution_map, predicted_class, confidence
    
    def get_top_regions(self, attribution_map, top_k=5):
        """
        Find the top-k most important regions in the attribution map.
        
        Args:
            attribution_map: Attribution scores
            top_k: Number of top regions to return
            
        Returns:
            tuple: (indices of top regions, corresponding importance scores)
        """
        # Compute absolute values
        abs_attr = np.abs(attribution_map)
        flat_attr = abs_attr.reshape(-1)
        
        # Get top-k indices and scores
        top_k_idx = np.argsort(flat_attr)[-top_k:]
        importance_scores = flat_attr[top_k_idx]
        
        return top_k_idx, importance_scores

# Example usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader, Subset
    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # url = "http://farm3.staticflickr.com/2066/1798910782_5536af8767_z.jpg"
    # url = "http://farm1.staticflickr.com/184/399924547_98e6cef97a_z.jpg"
    url = "http://farm1.staticflickr.com/128/318959350_1a39aae18c_z.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values.to(device)
    
    # Initialize analyzer with wrapped model
    analyzer = GradShapAnalyzer(
        model=pred_model,
        device=device
    )
    
    # Get attribution for single image
    attribution_map, pred_class, confidence = analyzer.get_attribution(pixel_values)
    
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.3f}")