import torch 
import torch.nn as nn
from maskgen.utils import idx_to_selector
import torch.nn.functional as F
import numpy as np
from maskgen.models import MLP
import math
from transformers import DistilBertModel, BertModel
from typing import List


class SimilarityMeasure(nn.Module):
    def __init__(self, pred_hidden_size, explain_hidden_size, embed_size=512):
        super(SimilarityMeasure, self).__init__()

        self.pred_map = MLP(pred_hidden_size, 128, embed_size, num_blocks=2, bottleneck_dim=64)
        self.explain_map = MLP(explain_hidden_size, 128, embed_size, num_blocks=2, bottleneck_dim=64)

        self.logit_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, pred_feature, explain_features):
        """
        Forward pass of the model.

        Args:
            q (torch.Tensor): Query tensor of shape [N, pred_hidden_size].
            k (torch.Tensor): Key tensor of shape [N, L, explain_hidden_size].

        Returns:
            torch.Tensor: Similarity tensor of shape [N, L].
        """
        pred_feature = F.normalize(self.pred_map(pred_feature), p=2, dim=-1).unsqueeze(1)  # [N, 1, embed_size]
        explain_features = F.normalize(self.explain_map(explain_features), p=2, dim=-1)  # [N, L, embed_size]


        logit_scale = self.logit_scale.exp()

        similarity = torch.matmul(explain_features, pred_feature.transpose(-1, -2)).squeeze(-1) * logit_scale  # [N, L]

        return similarity  # [N, L]


class MaskGeneratingModel(nn.Module):
    def __init__(self, pred_model: nn.Module, hidden_size, num_classes):
        super().__init__()

        # self.explain_model = DistilBertModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.explain_model = BertModel.from_pretrained("textattack/bert-base-uncased-SST-2")
        self.pred_model = pred_model
        self.num_classes = num_classes

        explain_hidden_size = self.explain_model.config.hidden_size
        self.hidden_size = hidden_size
        self.similarity_measure = SimilarityMeasure(hidden_size, explain_hidden_size)

        self.mask_token = 103

        self.bce_loss = nn.BCELoss(reduction='none')

        self.freeze_params()


    def freeze_params(self):
        """
        Freezes the parameters of the ViT and prediction model.

        This method sets the `requires_grad` attribute of all parameters in the ViT and prediction model to False,
        effectively freezing them and preventing them from being updated during training.
        """
        for param in self.explain_model.parameters():
            param.requires_grad = False
        for param in self.pred_model.parameters():
            param.requires_grad = False
    
    def get_interpretable_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Extract interpretable features using the text part of CLIP model.

        Args:
            x: An image tensor of shape [N, C, H, W].

        Returns:
            Interpretable features of shape [N, L, d].
        """
        # get the output of the ViT model
        # print(input_ids, attention_mask)
        output = self.explain_model(input_ids, attention_mask)
        # get the last hidden state, exclude the cls token
        hidden_states = output['last_hidden_state'] # [N, L, d], the first and last tokens are [CLS] and [SEP]
        return hidden_states
    
    def get_original_text_feature(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, predicted_class_selector: torch.Tensor):
        """
        Extract the original feature using the original prediction model.

        Args:
            input_ids: The input tensor of shape [N, L].

        Returns:
            Original feature of shape [N, d].
        """
        N = input_ids.shape[0]
        # get the output of the prediction model
        output = self.pred_model.bert(input_ids, attention_mask)
        # get the first hidden state, which corresponds to the cls token
        pooled_output = output[1] # [N, d]
        # pooled_output = hidden_state[:, 0, :] # [N, d]
        # pooled_output = self.pred_model.pre_classifier(pooled_output)  # (bs, dim)
        # pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)

        W = self.pred_model.classifier.weight # [n_classes, d]
        # original_feature = pooled_output.unsqueeze(1) * W.unsqueeze(0)   # [N, n_classes, d]
        original_feature = W.unsqueeze(0).expand(N, -1, -1) # [N, n_classes, d]
        original_feature = (original_feature * predicted_class_selector.unsqueeze(-1)).sum(1) # [N, d]
        return original_feature


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, predicted_class_selector: torch.Tensor):
        original_feature = self.get_original_text_feature(input_ids, attention_mask, predicted_class_selector) # [N, d]

        interpretable_features = self.get_interpretable_text_features(input_ids, attention_mask) # [N, L, d]

        
        sim = self.similarity_measure(pred_feature=original_feature, explain_features=interpretable_features) # [N, L]
        return {'sim': sim}
    

    def get_predicted_class_selector(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
            """
            Returns the predicted class selector for the given input tensor, with the predicted class set to 1 and all other classes set to 0.

            Args:
                x (torch.Tensor): The input tensor of shape [N, C, H, W].

            Returns:
                torch.Tensor: The predicted class selector tensor of shape [N, C], where N is the batch size and C is the number of classes.
            """
            logits = self.pred_model(input_ids, attention_mask).logits # [N, n_classes]
            predicted_class_idx = logits.argmax(-1) # [N, 1]
            predicted_class_selector = idx_to_selector(predicted_class_idx, self.num_classes) # [N, n_classes]
            return predicted_class_selector


    def loss_func(self, new_sim, old_sim, attention_mask, mask_list, masked_input_probs_list):
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
        L = new_sim.shape[-1]

        # generating probability
        mask_prob = torch.sigmoid(new_sim).unsqueeze(1).expand(-1, n_steps, -1) # [N, n_steps, L]
        old_mask_prob = torch.sigmoid(old_sim).unsqueeze(1).expand(-1, n_steps, -1).detach() # [N, n_steps, L]
        # reverse_mask_prob = 1 - mask_prob # [N, n_steps, L]
        # generated mask samples
        mask_samples = torch.stack(mask_list, dim=1) # [N, n_steps, L]

        # the prediction probability of the masked input
        mask_sample_probs = torch.stack(masked_input_probs_list, dim=1) # [N, n_steps, 1]
        # the prediction probability of the reverse masked input

        # reward loss, if mask_sample_probs is higher, we want to optimize the probability of generating the masks

        # reward = (torch.clamp(mask_sample_probs/(true_prob.unsqueeze(1) + 1e-5), 0.7, 1.3) -0.7) / 0.6
        reward = torch.relu(mask_sample_probs - 0.5) # [N, 1, 1]
        # reward = 0.3 - torch.abs( mask_sample_probs - true_prob.unsqueeze(1)) # [N, n_steps, 1]
        # prob_loss = torch.exp(-bce_loss(mask_prob , mask_samples).mean(-1, keepdim=True)) #* mask_samples # [N, n_steps, L]
        # prob_loss = bce_loss(mask_prob , mask_samples) #* mask_samples # [N, n_steps, L]
        old_logprob = -bce_loss(old_mask_prob, mask_samples).sum([-1], keepdims=True) # [N, 1, 1]
        new_logprob = -bce_loss(mask_prob, mask_samples).sum([-1], keepdims=True) # [N, 1, 1]
        # prob_loss = (mask_prob / old_mask_prob).log() * mask_samples + ((1 - mask_prob) / (1 - old_mask_prob)).log() * (1 - mask_samples) # [N, n_steps, L]
        ratio = (new_logprob - old_logprob).exp() # [N, 1, 1]
        # print(ratio[:,0,0].view(-1))
        surr1 = ratio * reward # [N, 1, 1]
        surr2 = torch.clamp(ratio, 0.7, 1.3) * reward # [N, 1, 1]
        reward_loss = -torch.min(surr1, surr2).mean() # [N, ]
        # print('surr1', surr1.mean())
        # print('surr2', surr2.mean())
        # print('reward_loss', reward_loss)
        # prob_loss = prob_loss.sum(-1, keepdim=True) # [N, n_steps, 1]
        # prob_loss = torch.exp(prob_loss) # [N, n_steps, 1]
        # reward_loss = - (prob_loss * reward).mean() # [N, n_steps, L]

        # mask_loss
        # mask_loss = torch.abs(1 - mask_prob.mean([-1, -2]) - mask_sample_probs.mean([-1, -2])).mean() 
        # mask_loss = ((0.5 - mask_prob.mean([1, 2]))**2).mean()  
        mask_loss = (mask_prob.mean([1]) * attention_mask).sum(-1) / attention_mask.sum(-1)  # [N, ] 
        mask_loss = ((0.8 - mask_loss - mask_sample_probs.mean([-1, -2]))**2)
        mask_loss = mask_loss.mean()

        loss =  reward_loss  + 1* mask_loss
        mask_mean = mask_prob.mean([1, 2]) # [N]
        prob_mean = mask_sample_probs.mean([1, 2]) # [N]

        return {'loss': loss,
                'reward_loss': reward_loss,
                'mask_mean': mask_mean.mean(),
                'prob_mean': prob_mean.mean(),
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
            # scaler = torch.rand(sim.shape[0], 1, device=sim.device) 

            # mask_prob = (mask_prob - 0) / (1e-5 + mask_prob.max(dim=-1, keepdim=True)[0])
            # mask_prob = mask_prob * scaler
            # mask_prob = torch.clamp(mask_prob, 0.2, 1.0) # prevent the mask_prob from being too close to 0 or 1

            # sample a mask (action) based on the mask probability (policy)
            mask = torch.bernoulli(mask_prob) # [N, L]
        
        return mask # [N, L]
    
    def get_mask_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, mask: torch.Tensor, predicted_class_selector: torch.Tensor):
        # No gradients upon the parameters of the prediction model
        with torch.no_grad():
            masked_input_ids = input_ids * mask + (1 - mask) * self.mask_token  # [N, L]
            masked_input_ids = masked_input_ids.long()

            masked_probs = torch.softmax(self.pred_model(masked_input_ids, attention_mask).logits, dim=-1) # [N, n_classes]
            masked_probs = (masked_probs * predicted_class_selector).sum(-1, keepdim=True) # [N, 1]

        return masked_probs


    def sample_one_step(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, sim: torch.Tensor, predicted_class_selector: torch.Tensor):
        with torch.no_grad():
            mask = self.generate_mask(sim)
            mask_probs = self.get_mask_probs(input_ids, attention_mask, mask, predicted_class_selector)
            # reverse_mask = 1.0 - mask
            # reverse_mask_probs = self.get_mask_probs(x, reverse_mask, predicted_class_selector)
        return mask, mask_probs #, reverse_mask, reverse_mask_probs


    def train_one_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, optimizer: torch.optim.Optimizer, n_steps=10, n_samples=1):
        device = input_ids.device
        self.train()
        # save old state dicts to calculate reference p'
        old_state_dicts = self.state_dict()
        old_model = MaskGeneratingModel(self.pred_model, self.hidden_size, self.num_classes).to(device)
        old_model.load_state_dict(old_state_dicts)
        
        predicted_class_selector = self.get_predicted_class_selector(input_ids, attention_mask)
        # get the sim tensor of the old model
        old_sim = old_model.forward(input_ids, attention_mask, predicted_class_selector)['sim']

        mask_list = []
        masked_probs_list = []
        # update for n steps
        for _ in range(n_steps):
            optimizer.zero_grad()
            for _ in range(n_samples):
                # get the mask and masked inputs' output probs generated from the old sim tensor
                mask, masked_probs = self.sample_one_step(input_ids, attention_mask, old_sim, predicted_class_selector)
                mask_list.append(mask)
                masked_probs_list.append(masked_probs)
            # get the sim tensor of the new model
            new_sim = self.forward(input_ids, attention_mask, predicted_class_selector)['sim']
            loss_dict = self.loss_func(new_sim, old_sim, attention_mask, mask_list, masked_probs_list)

            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
        return loss_dict


    def train_one_batch_old(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, optimizer: torch.optim.Optimizer, n_steps=10):
        self.train()
        optimizer.zero_grad()
        predicted_class_selector = self.get_predicted_class_selector(input_ids, attention_mask)
        outputs = self.forward(input_ids, attention_mask, predicted_class_selector)
        sim = outputs['sim']
        

        mask_list, reverse_mask_list = [], []
        masked_probs_list, reverse_masked_probs_list = [], []
        for idx in range(n_steps):

            mask, masked_probs = self.sample_one_step(input_ids, attention_mask, sim, predicted_class_selector)
            reverse_mask, reverse_masked_probs = self.sample_one_step(input_ids, attention_mask, -sim, predicted_class_selector)

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

    


    
    def attribute_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, label=None):
        with torch.no_grad():
            if label is None:
                predicted_class_selector = self.get_predicted_class_selector(input_ids, attention_mask)
            else:
                predicted_class_selector = idx_to_selector(label, self.num_classes)
            outputs = self.forward(input_ids, attention_mask, predicted_class_selector)
            probs = torch.sigmoid(outputs['sim'])
        return probs
    