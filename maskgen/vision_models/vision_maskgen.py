import torch 
import torch.nn as nn
from maskgen.utils import idx_to_selector
import torch.nn.functional as F
import numpy as np
from maskgen.models import MLP
import math
from transformers import ViTModel, CLIPVisionModel
from typing import List
from transformers.models.bert.modeling_bert import BertEncoder, BertPredictionHeadTransform
from transformers import BertConfig 


class Encoder(BertEncoder):
    def __init__(self, hidden_size=768, intermediate_size=3072):
        config = BertConfig()
        config.num_hidden_layers = 6
        config.hidden_size = hidden_size
        config.intermediate_size = 3072
        config.num_attention_heads = 12
        super().__init__(config)
        self.transform = BertPredictionHeadTransform(config)
    
    def forward(self, hidden_states):
        last_hidden_state = super().forward(hidden_states)['last_hidden_state']
        output = self.transform(last_hidden_state)
        return output


class SimilarityMeasure(nn.Module):
    def __init__(self, hidden_size, intermediate_size=512):
        super(SimilarityMeasure, self).__init__()
        self.pred_map = Encoder(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, feature):
        """
        Forward pass of the model.

        Args:
            feature (torch.Tensor): Feature tensor of shape [N, L+1, hidden_size].

        Returns:
            torch.Tensor: Similarity logits of shape [N, L].
        """
        feature = self.pred_map(feature)

        logits = self.fc(feature).squeeze(-1) # [N, L+1]

        return logits[:, 1:]  # [N, L]


class MaskGeneratingModel(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.similarity_measure = SimilarityMeasure(hidden_size=hidden_size)

        self.bce_loss = nn.BCELoss(reduction='none')

    @torch.no_grad()
    def get_features(self, model, x: torch.Tensor):
        """
        Extract the features from the given model.
        
        """
        # get the output of the ViT model in the classifier
        output = model.vit(x)
        # get the last hidden state
        hidden_states = output['last_hidden_state'] # [N, L+1, d]
        return hidden_states


    def forward(self, model, pixel_values: torch.Tensor):
        hidden_states = self.get_features(model, pixel_values)
        
        sim = self.similarity_measure(hidden_states) # [N, L]
        return sim

    @torch.no_grad()
    def generate_mask(self, sim):
        """Generate a mask based on the similarity tensor. [generate action based on policy]

        Args:
            sim (Tensor): The similarity tensor of shape [N, L].

        Returns:
            Tensor: The generated mask tensor of shape [N, L].
        """
        mask_prob = torch.sigmoid(sim)
        mask = torch.bernoulli(mask_prob) # [N, L]
        
        return mask # [N, L]

    @torch.no_grad()
    def get_pred_logits(self, model, pixel_values: torch.Tensor, mask=None):
        """
        Get the prediction probabilities of the masked input.
        args:
            model: The model used for prediction.
            pixel_values: (N, L, H, W) The pixel values of the input image.
            mask: (N, L) The mask tensor.
        output:
            masked_pred_logits: (N, num_classes) The prediction probabilities of the masked inputs.
        """
        if mask is not None:
            bool_masked_pos = 1.0 - mask # [N, L] 1 for masked, 0 for unmasked
            pred_logits = model(pixel_values, bool_masked_pos=bool_masked_pos).logits # [N, num_classes]
        else:
            pred_logits = model(pixel_values).logits

        return pred_logits


    @torch.no_grad()
    def sample_one_step(self, model, pixel_values: torch.Tensor, sim: torch.Tensor):
        mask = self.generate_mask(sim)
        masked_pred_logits = self.get_pred_logits(model, pixel_values, mask)
        return mask, masked_pred_logits
    
    def calculate_reward(self, masked_pred_logits, pred_logits):
        """
        Calculate the reward for the given mask.

        Args:
            pred_logits (Tensor): The original input prediction logits tensor of shape [N, num_classes].
            mask_pred_logits (Tensor): The masked input prediction logits tensor of shape [N, num_classes].

        Returns:
            Tensor: The reward tensor of shape [N, ].
        """
        reward = self.bce_loss(torch.sigmoid(masked_pred_logits), torch.sigmoid(pred_logits)) # [N, num_classes]
        reward = 1 / (reward + 1e-5) # [N, num_classes]
        reward = (reward * torch.sigmoid(pred_logits)).mean(-1) # [N,]
        return reward


    def loss_func(self, sim, mask, masked_pred_logits, pred_logits):
        """Calculate the loss for the given mask.
        Args:
            sim: The similarity tensor of shape [N, L].
            mask: The mask tensor of shape [N, L].
            masked_pred_logits: The prediction logits of the masked input. [N, num_classes]
            pred_logits: The prediction logits of the original input. [N, num_classes]
        """
        reward = self.calculate_reward(masked_pred_logits, pred_logits) # [N,]
        fit_loss = self.bce_loss(torch.sigmoid(sim), mask).mean(-1) # [N,]
        reward_loss = (reward * fit_loss).mean()

        # mask loss
        # TODO: here we use softmax to enforce the ranking nature of the explanation.
        mask_loss = (mask * torch.softmax(sim)).mean(-1) # [N,]
        mask_loss = mask_loss.mean()

        # total loss
        loss = reward_loss + mask_loss

        # other outputs
        mask_mean = mask.mean()
        prob_mean = torch.sigmoid(sim).mean()



        return {'loss': loss,
                'reward_loss': reward_loss,
                'mask_loss': mask_loss,
                'mask_mean': mask_mean,
                'prob_mean': prob_mean,
                }

    
    def train_one_batch(self, x: torch.Tensor, optimizer: torch.optim.Optimizer, n_steps=10):
        self.train()
        optimizer.zero_grad()
        predicted_class_selector = self.get_predicted_class_selector(x)
        outputs = self.forward(x, predicted_class_selector)
        sim = outputs['sim']
        

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
        # loss_dict = self.loss_func(sim, mask_list, masked_probs_list)

        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()
        return loss_dict

    

    



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
            predicted_class_selector = self.get_predicted_class_selector(x)
            outputs = self.forward(x, predicted_class_selector)
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
    