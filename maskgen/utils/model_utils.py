from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from maskgen.vision_models.vision_maskgen import MaskGeneratingModel

def get_pred_model(pretrained_name, device):
    """Initialize and load the prediction model"""
    
    # Load processor and initialize prediction model
    processor = ViTImageProcessor.from_pretrained(pretrained_name)
    pred_model = ViTForImageClassification.from_pretrained(pretrained_name)
    pred_model.to(device)
    pred_model.eval()
    
    return processor, pred_model


def load_exp_and_target_model(config, device):
    # Load Model
    pretrained_name = config['pretrained_name']
    processor, target_model = get_pred_model(pretrained_name, device)
    vit_config = target_model.config

    # Target model for explanation
    target_model = ViTForImageClassification.from_pretrained(pretrained_name)
    target_model.eval()
    target_model.to(device)

    # Load trained weights
    maskgen_model = MaskGeneratingModel.load_model(base_model_name=pretrained_name, 
                                    save_path=config['model_path'], 
                                    hidden_size=vit_config.hidden_size, 
                                    num_classes=vit_config.num_labels)
    maskgen_model.eval()
    maskgen_model.to(device)

    return target_model, maskgen_model, processor