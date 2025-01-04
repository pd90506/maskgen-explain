# Refactored vision mask gen model. 
# Splited the model structure and the training logic 

from transformers import ViTModel
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
import torch.nn as nn
import re
import os
import shutil

def convert_to_peft(base_model: ViTModel, r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1) -> PeftModel:
    """Convert base model to PEFT model."""
    # Find target modules in base model
    target_modules = []
    for name, _ in base_model.named_modules():
        if re.match(r'.*(?:query|key|value|dense)$', name):
            target_modules.append(name)
            
    if not target_modules:
        raise ValueError("No matching target modules found in base model")
        
    # Configure LoRA
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules
    )
    
    # Convert to PEFT
    peft_model = get_peft_model(base_model, lora_config)
    print(f"Converted base model layers to PEFT: {target_modules}")
    return peft_model


class MaskGeneratingModel(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_size: int, num_classes: int, freeze_base: bool = False):
        super().__init__()

        self.base_model = base_model
        self.num_classes = num_classes

        # Actor network for policy generation
        self.actor = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

        # Critic network for value estimation
        self.critic = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  # Outputs a single value
        )

        # Freeze the base model if required
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor):
        """
        Forward pass through the model.

        Args:
            pixel_values (torch.Tensor): Input image tensor of shape (N, C, H, W).

        Returns:
            dist (torch.distributions.Bernoulli): Bernoulli distribution for mask generation.
            value (torch.Tensor): Critic value of shape (N,).
            mu_logits (torch.Tensor): Logits generated by the actor of shape (N, L, num_classes).
        """
        # Forward pass through the base model
        base_output = self.base_model(pixel_values)
        hidden_states = base_output['last_hidden_state']  # Shape: [N, L+1, hidden_size]

        # Actor outputs logits for all tokens
        mu_logits = self.actor(hidden_states[:, 1:, :])  # Shape: [N, L, num_classes]

        # Critic value estimation using the [CLS] token
        pooled_output = hidden_states[:, 0, :]  # [CLS] token representation
        value = self.critic(pooled_output) # Shape: [N, 1]

        return mu_logits, value
    
    def get_dist_critic(self, pixel_values: torch.Tensor, labels: torch.Tensor):
        """
        Get the distribution, critic value, and the token-wise sum of probabilities for each class.

        Args:
            pixel_values (torch.Tensor): Input image tensor of shape (N, C, H, W).
            labels (torch.Tensor): Provided labels of shape (N, 1).

        Returns:
            dist (torch.distributions.Bernoulli): Bernoulli distribution for mask generation.
            true_value (torch.Tensor): Critic value for the true label of shape (N, 1).
            mu_sum_prob (torch.Tensor): token-wise sum of probabilities for each class of shape (N, num_classes).
        """
        mu_logits, value = self(pixel_values)
        # expand the labels to the same shape as mu_logits
        labels_expanded = labels.unsqueeze(1).expand(-1, mu_logits.shape[1], -1) # [N, L, 1]
        mu_true_logits = torch.gather(mu_logits, -1, labels_expanded).squeeze(-1) # [N, L]
        # true_value = torch.gather(value, -1, labels) # [N, 1]
        true_value = value

        dist = torch.distributions.Bernoulli(logits=mu_true_logits)

        # Calculate the mean of probabilities of each class for each token
        mu_prob = torch.softmax(mu_logits, -1) # [N, L, num_classes]
        mu_mean_prob = torch.mean(mu_prob, dim=1) # [N, num_classes]
        return dist, true_value, mu_mean_prob


    @torch.no_grad()
    def attribute_img(self, pixel_values: torch.Tensor, labels: torch.Tensor, image_size: int = 224, patch_size: int = 16):
        """
        Generate attribution heatmaps for input images.

        Args:
            pixel_values (torch.Tensor): Input image tensor of shape (N, C, H, W).
            labels (torch.Tensor): Ground truth labels of shape (N,).
            image_size (int, optional): Size of the input image (H = W = image_size). Defaults to 224.
            patch_size (int, optional): Size of the patch in ViT. Defaults to 16.

        Returns:
            heatmap (torch.Tensor): Attribution heatmap of shape (N, H/patch_size, W/patch_size).
        """
        dist, _, _ = self.get_dist_critic(pixel_values, labels)
        sim_probs = dist.probs  # Shape: [N, L]

        grid_size = image_size // patch_size
        heatmap = sim_probs.view(-1, grid_size, grid_size)  # Shape: [N, grid_size, grid_size]
        return heatmap

    def save_model(self, save_path: str):
        """Save PEFT adapters and other model components."""
        # Check if directory exists, else create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Save PEFT adapters only
        if isinstance(self.base_model, PeftModel):
            self.base_model.save_pretrained(save_path)
        
        # Save actor and critic
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, f"{save_path}/other_components.pt")


    @classmethod
    def load_model(cls, base_model_name: str, save_path: str, hidden_size: int, num_classes: int):
        """Load model with PEFT adapters."""
        # First load the base model
        base_model = ViTModel.from_pretrained(base_model_name)
        
        # Convert to PEFT and load adapters
        peft_base_model = convert_to_peft(base_model)
        peft_base_model = PeftModel.from_pretrained(
            peft_base_model,
            save_path
        )
        
        # Initialize model with loaded base
        model = cls(peft_base_model, hidden_size, num_classes, freeze_base=False)
        
        # Load actor and critic
        components = torch.load(f"{save_path}/other_components.pt")
        model.actor.load_state_dict(components['actor_state_dict'])
        model.critic.load_state_dict(components['critic_state_dict'])
        
        return model

def main():
    # Define parameters
    base_model_name = 'google/vit-base-patch16-224'
    save_path = './test_saved_model'
    hidden_size = 512
    num_classes = 10

    # Create a base model
    base_model = ViTModel.from_pretrained(base_model_name)

    # Convert to PEFT model
    peft_model = convert_to_peft(base_model)

    # Initialize MaskGeneratingModel
    model = MaskGeneratingModel(peft_model, hidden_size, num_classes)

    # Dummy inputs for testing
    pixel_values = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    labels = torch.randint(0, num_classes, (2,))  # Random labels

    # Forward pass
    mu_logits, value = model(pixel_values, labels)
    print("Forward pass outputs:", mu_logits.shape, value.shape)

    # Save the model
    model.save_model(save_path)
    print(f"Model saved to {save_path}")

    # Load the model
    loaded_model = MaskGeneratingModel.load_model(base_model_name, save_path, hidden_size, num_classes)
    print("Model loaded successfully")

    # Verify loaded model
    mu_logits_loaded, value_loaded = loaded_model(pixel_values, labels)
    print("Loaded model outputs:", mu_logits_loaded.shape, value_loaded.shape)

    # Clean up
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        print(f"Cleaned up directory: {save_path}")

if __name__ == "__main__":
    main()