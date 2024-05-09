import torch 
import torch.nn as nn
from maskgen.utils import idx_to_selector
import torch.nn.functional as F
import numpy as np
from maskgen.models import MLP
import math
from transformers import ViTModel, CLIPVisionModel
from typing import List


class SimilarityMeasure(nn.Module):
    def __init__(self, pred_hidden_size, explain_hidden_size, embed_size=512):
        super(SimilarityMeasure, self).__init__()
        self.query = nn.Linear(pred_hidden_size, embed_size)
        self.key = nn.Linear(explain_hidden_size, embed_size)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, q, k):
        """
        Forward pass of the model.

        Args:
            q (torch.Tensor): Query tensor of shape [N, pred_hidden_size].
            k (torch.Tensor): Key tensor of shape [N, L, explain_hidden_size].

        Returns:
            torch.Tensor: Similarity tensor of shape [N, L].
        """
        Q = F.normalize(self.query(q), p=2, dim=-1)  # [N, embed_size]
        K = F.normalize(self.key(k), p=2, dim=-1)  # [N, L, embed_size]
        logit_scale = self.logit_scale.exp()

        similarity = torch.matmul(K, Q.unsqueeze(-1)).squeeze(1).squeeze(-1) * logit_scale  # [N, L]

        return similarity  # [N, L]


class MaskGeneratingModel(nn.Module):
    def __init__(self, pred_model: nn.Module, hidden_size, num_classes):
        super().__init__()

        self.vit = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14')
        self.pred_model = pred_model
        self.num_classes = num_classes

        explain_hidden_size = self.vit.config.hidden_size
        pred_hidden_size = hidden_size
        self.similarity_measure = SimilarityMeasure(pred_hidden_size, explain_hidden_size)

        self.bce_loss = nn.BCELoss(reduction='none')

        self.freeze_params()


    def freeze_params(self):
        """
        Freezes the parameters of the ViT and prediction model.

        This method sets the `requires_grad` attribute of all parameters in the ViT and prediction model to False,
        effectively freezing them and preventing them from being updated during training.
        """
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.pred_model.parameters():
            param.requires_grad = False
    
    def get_interpretable_vision_features(self, x: torch.Tensor):
        """
        Extract interpretable features using the ViT part of CLIP model.

        Args:
            x: An image tensor of shape [N, C, H, W].

        Returns:
            Interpretable features of shape [N, L, d].
        """
        # get the output of the ViT model
        output = self.vit(x)
        # get the last hidden state, exclude the cls token
        hidden_states = output['last_hidden_state'][:, 1:, :]  # [N, L, d]
        return hidden_states
    
    def get_original_vision_feature(self, x: torch.Tensor):
        """
        Extract the original feature using the original prediction model.

        Args:
            x: An image tensor of shape [N, C, H, W].

        Returns:
            Original feature of shape [N, d].
        """
        # get the output of the ViT model
        output = self.pred_model.vit(x)
        # get the first hidden state, which corresponds to the cls token
        hidden_state = output[0][:, 0, :] # [N, d]
        return hidden_state

    def get_predicted_class_selector(self, x: torch.Tensor):
            """
            Returns the predicted class selector for the given input tensor, with the predicted class set to 1 and all other classes set to 0.

            Args:
                x (torch.Tensor): The input tensor of shape [N, C, H, W].

            Returns:
                torch.Tensor: The predicted class selector tensor of shape [N, C], where N is the batch size and C is the number of classes.
            """
            logits = self.pred_model(x).logits # [N, n_classes]
            predicted_class_idx = logits.argmax(-1) # [N, 1]
            predicted_class_selector = idx_to_selector(predicted_class_idx, self.num_classes) # [N, n_classes]
            return predicted_class_selector


    def loss_func(self, sim, mask_list, reverse_mask_list, masked_prob_list, reverse_masked_prob_list):
        """Calculate the loss for the given mask.

        Args:
            sim (Tensor): The similarity tensor of shape [N, L].
            mask (Tensor): The generated mask tensor of shape [N, L].
            probs (Tensor): The probability tensor of shape [N,]. Obtained by feeding the randomly generated mask to the prediction model.

        Returns:
            Tensor: The loss tensor of shape [N, L].
        """
        bce_loss = nn.BCELoss(reduction='none')
        n_steps = len(mask_list)
        L = sim.shape[-1]

        # generating probability
        mask_prob = torch.sigmoid(sim).unsqueeze(1).expand(-1, n_steps, -1) # [N, n_steps, L]
        reverse_mask_prob = 1 - mask_prob # [N, n_steps, L]
        # generated mask samples
        mask_samples = torch.stack(mask_list, dim=1) # [N, n_steps, L]
        reverse_mask_samples = torch.stack(reverse_mask_list, dim=1) # [N, n_steps, L]

        # the prediction probability of the masked input
        mask_sample_probs = torch.stack(masked_prob_list, dim=1) # [N, n_steps, 1]
        # the prediction probability of the reverse masked input
        reverse_mask_sample_probs = torch.stack(reverse_masked_prob_list, dim=1) # [N, n_steps, 1]

        # reward loss, if mask_sample_probs is higher, we want to optimize the probability of generating the masks
        reward_loss = bce_loss(mask_prob, mask_samples) # [N, n_steps, L]
        reward_loss = (reward_loss * mask_sample_probs).sum() / (L * (mask_sample_probs + 0.1).sum())  # [N, n_steps, L]

        # regret loss, if reverse_mask_sample_probs is higher, we want to optimize the probability of generating the reverse masks
        # reverse_mask_samples = torch.clamp(mask_samples + reverse_mask_samples, 0.0, 1.0)
        regret_loss = bce_loss(mask_prob, reverse_mask_samples) # [N, n_steps, L]
        regret_loss = (regret_loss * reverse_mask_sample_probs).sum() / (L * (reverse_mask_sample_probs + 0.1).sum()) # [N, n_steps, L]
        # # total_reward_loss = reward_loss.sum() / (mask_selector.sum() + 1e-5)  + regret_loss.sum() / ((1 - mask_selector).sum() + 1e-5) # [1]


        # mask_loss
        mask_loss = mask_prob.mean()


        loss =  reward_loss + 0.1 * regret_loss + 0.1 * mask_loss
        return {'loss': loss,
                'reward_loss': reward_loss,
                'regret_loss': regret_loss,
                'mask_loss': mask_loss}

    def generate_mask(self, sim):
        """Generate a mask based on the similarity tensor. [generate action based on policy]

        Args:
            sim (Tensor): The similarity tensor of shape [N, L].

        Returns:
            Tensor: The generated mask tensor of shape [N, L].
        """
        with torch.no_grad():
            mask_prob = torch.sigmoid(sim)

            # sample a mask (action) based on the mask probability (policy)
            mask = torch.bernoulli(mask_prob) # [N, L]
        
        return mask # [N, L]
    
    def get_mask_probs(self, x: torch.Tensor, mask: torch.Tensor, predicted_class_selector: torch.Tensor):
        # No gradients upon the parameters of the prediction model
        with torch.no_grad():
            H, W = x.shape[-2:]
            mask_size = int(math.sqrt(mask.shape[-1]))
            reshaped_mask = mask.reshape(-1, mask_size, mask_size).unsqueeze(1) # [N, 1, size, size]
            upsampled_mask = F.interpolate(reshaped_mask, size=(H, W), mode='nearest') # [N, 1, H, W]
            masked_input = x * upsampled_mask # [N, C, H, W]
            masked_probs = torch.softmax(self.pred_model(masked_input).logits, dim=-1) # [N, n_classes]
            masked_probs = (masked_probs * predicted_class_selector).sum(-1, keepdim=True) # [N, 1]

        return masked_probs

    def sample_one_step(self, x: torch.Tensor, sim: torch.Tensor, predicted_class_selector: torch.Tensor):
        with torch.no_grad():
            mask = self.generate_mask(sim)
            mask_probs = self.get_mask_probs(x, mask, predicted_class_selector)
            # reverse_mask = self.generate_mask(-sim)
            # reverse_mask_probs = self.get_mask_probs(x, reverse_mask, predicted_class_selector)
        return mask, mask_probs
    
    def train_one_batch(self, x: torch.Tensor, optimizer: torch.optim.Optimizer, n_steps=10):
        self.train()
        optimizer.zero_grad()
        outputs = self.forward(x)
        sim = outputs['sim']
        predicted_class_selector = self.get_predicted_class_selector(x)

        mask_list, reverse_mask_list = [], []
        masked_probs_list, reverse_masked_probs_list = [], []
        for idx in range(n_steps):

            mask, masked_probs = self.sample_one_step(x, sim, predicted_class_selector)
            reverse_mask, reverse_masked_probs = self.sample_one_step(x, -sim, predicted_class_selector)

            mask_list.append(mask)
            reverse_mask_list.append(reverse_mask)
            masked_probs_list.append(masked_probs)
            reverse_masked_probs_list.append(reverse_masked_probs)
        
        loss_dict = self.loss_func(sim, mask_list, reverse_mask_list, masked_probs_list, reverse_masked_probs_list)

        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()
        return loss_dict

    
    def forward(self, x: torch.Tensor):
        interpretable_features = self.get_interpretable_vision_features(x)

        original_feature = self.get_original_vision_feature(x) # [N, d]
        
        sim = self.similarity_measure(q=original_feature, k=interpretable_features) # [N, L]
        return {'sim': sim}
    



    def attribute_img(self, 
                      x, 
                      image_size=224, 
                      patch_size=16, 
                      baseline=None, 
                      seed=None):
        """
        Generate attribution heatmap for an input image.

        Args:
            x: An image tensor of shape [N, C, H, W], where H = W = image_size.
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

        size = image_size // patch_size
        N, C, H, W = x.shape
        with torch.no_grad():
            outputs = self.forward(x)
            # mask_list = outputs['mask_list']
            # probs_list = outputs['probs_list']
            probs = torch.sigmoid(outputs['sim']).reshape(N, size, size)
           
            # mask = torch.stack(mask_list, dim=1) # [N, n_samples, L]
            # probs = torch.stack(probs_list, dim=1) # [N, n_samples, 1]
            # weighted_mask = (mask * probs).sum(1) # [N, L]
            # weighted_mask = weighted_mask.reshape(N, size, size)
        
        return probs
    
    def attribute_text(self, x):
        # TODO
        raise NotImplementedError("This function hasn't been developed.")