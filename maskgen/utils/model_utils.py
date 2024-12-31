from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig

def get_pred_model(pretrained_name, device):
    """Initialize and load the prediction model"""
    
    # Load processor and initialize prediction model
    processor = ViTImageProcessor.from_pretrained(pretrained_name)
    pred_model = ViTForImageClassification.from_pretrained(pretrained_name)
    pred_model.to(device)
    pred_model.eval()
    
    return processor, pred_model