from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    ViTModel,
    ViTConfig,
)


def get_pred_model(pretrained_name, device):
    """Initialize and load the prediction model"""
    
    # Load configuration and processor
    model_config = ViTConfig.from_pretrained(pretrained_name)
    processor = ViTImageProcessor.from_pretrained(pretrained_name)
    
    # Initialize prediction model
    pred_model = ViTForImageClassification.from_pretrained(pretrained_name)
    pred_model.to(device)
    pred_model.eval()
    
    return processor, pred_model