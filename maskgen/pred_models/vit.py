import torch
from PIL import Image
import requests
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    ViTConfig,
)

def load_image_from_url(url):
    """
    Load an image from a given URL.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure the request was successful
    return Image.open(response.raw)

def load_vit_model(pretrained_name, device):
    """
    Load a pre-trained ViT model and its associated image processor.
    """
    config = ViTConfig.from_pretrained(pretrained_name)
    processor = ViTImageProcessor.from_pretrained(pretrained_name)
    model = ViTForImageClassification.from_pretrained(pretrained_name)
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model, processor

def predict_image_class(image, model, processor, device):
    """
    Predict the class of an input image using the provided ViT model.
    """
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        return model.config.id2label[predicted_class_idx]

def main():

    from maskgen.utils import get_device
    # Setup device
    device = get_device()

    # Image URL
    url = "http://farm1.staticflickr.com/128/318959350_1a39aae18c_z.jpg"
    image = load_image_from_url(url)

    # Pre-trained model name
    pretrained_name = 'google/vit-base-patch16-224'

    # Load the ViT model and processor
    model, processor = load_vit_model(pretrained_name, device)

    # Predict image class
    predicted_class = predict_image_class(image, model, processor, device)
    print("Predicted class:", predicted_class)

if __name__ == "__main__":
    main()
