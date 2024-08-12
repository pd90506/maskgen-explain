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
from torch.distributions import Bernoulli


class MaskGeneratingModel(nn.Module):
    def __init__(self, base_model:nn.Module, hidden_size, num_classes):
        super().__init__()
        self.bce_loss = nn.BCELoss(reduction='none')
        self.base_model = base_model
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.fc = nn.Linear(hidden_size, 1, bias=False)
        self.num_classes = num_classes
    
    
    def get_dist_critic(self, pixel_values):
        """
        Calculates the distribution of the mask and the critic value given the pixel values.
        Args:
            pixel_values (torch.Tensor) [N, H, W]: Tensor containing the pixel values.
        Returns:
            dist (torch.distributions.Bernoulli): Bernoulli distribution of mask.
            value (torch.Tensor): The critic value.
        """
        # print(pixel_values.shape)
        output = self.base_model(pixel_values)

        hidden_states = output['last_hidden_state'][:,1:,:]
        mu_logits = self.actor(hidden_states).squeeze(-1)
        dist = Bernoulli(logits=mu_logits)

        pooled_output = output['last_hidden_state'][:,0,:] # [N, d]
        value = self.critic(pooled_output) # [N, 1]
        # print("value", value.shape)
        return dist, value


    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        """
        Generates mini-batches of data for Proximal Policy Optimization (PPO) algorithm.
        Parameters:
            mini_batch_size (int): The size of each mini-batch.
            states (Tensor): The states of the environment.
            actions (Tensor): The actions taken in the environment.
            log_probs (Tensor): The logarithm of the probabilities of the actions taken.
            returns (Tensor): The expected returns for each state-action pair.
            advantage (Tensor): The advantage estimates for each state-action pair.
        Yields:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: A mini-batch of states, actions, log probabilities,
            expected returns, and advantage estimates.
        """
        
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
            
            
    def ppo_update(self, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        """
        Perform PPO (Proximal Policy Optimization) update for the given number of epochs.
        Parameters:
        - optimizer (torch.optim.Optimizer): The optimizer used for updating the model.
        - ppo_epochs (int): Number of PPO update epochs.
        - mini_batch_size (int): Size of mini-batches for PPO update.
        - states (Tensor): Input states.
        - actions (Tensor): Actions taken.
        - log_probs (Tensor): Log probabilities of actions.
        - returns (Tensor): Expected returns.
        - advantages (Tensor): Advantage estimates.
        - clip_param (float, optional): Clipping parameter for PPO update. Defaults to 0.2.
        """

        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                dist, value = self.get_dist_critic(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                # PPO step
                actor_loss  = - torch.min(surr1, surr2).mean()
                # learn the value function based on the estimated return
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.0001 * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Print the losses after one ppo update
        return  {"loss": loss.item(), 
                 "actor_loss": actor_loss.item(), 
                 "critic_loss": critic_loss.item(),
                 "returns": return_.mean().item(),
                 'entropy': entropy.item()}
    
    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        """
        Computes the Generalized Advantage Estimation (GAE) for the given rewards, masks, values, and next value.
        Parameters:
        - next_value (Tensor): The value of the next state.
        - rewards (Tensor): The rewards received.
        - masks (Tensor): The masks of the environment.
        - values (Tensor): The value estimates.
        - gamma (float, optional): The discount factor. Defaults to 0.99.
        - tau (float, optional): The GAE parameter. Defaults to 0.95.
        Returns:
        - list: List of GAE-estimated returns for each step.
        """
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def compute_gae_static_state(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        """
        Computes the Generalized Advantage Estimation (GAE) for a static state.
        Args:
            next_value (float): The value of the next state.
            rewards (list): List of rewards for each step.
            masks (list): List of masks indicating whether the episode has ended at each step.
            values (list): List of predicted values for each step.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            tau (float, optional): GAE parameter. Defaults to 0.95.
        Returns:
            list: List of GAE-estimated returns for each step.
        """

        values = np.append(values, next_value)
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] - (1 - gamma) * values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns
    
    @torch.no_grad()
    def get_action_reward(self, model, pixel_values, mask, dist):
        # obtain the generative distribution of the mask
        prob = torch.sigmoid(dist.logits)
        prob = F.normalize(prob, p=1, dim=-1)

        pred_logits = model(pixel_values).logits # [N, num_classes]
        pred = pred_logits.argmax(-1) # [N,]
        selector = idx_to_selector(pred, self.num_classes) # [N, num_classes]

        pred_prob = torch.softmax(pred_logits, -1) # [N, num_classes]
        pred_prob = (pred_prob * selector).sum(-1) # [N,]

        bool_masked_pos = 1.0 - mask
        masked_pred_logits = model(pixel_values, bool_masked_pos=bool_masked_pos).logits # [N, num_classes]
        masked_pred_prob = torch.softmax(masked_pred_logits, -1) # [N, num_classes]
        masked_pred_prob = (masked_pred_prob * selector).sum(-1) # [N,]

        reward = torch.exp(torch.log(masked_pred_prob) - torch.log(pred_prob)) # [N,]

        # mask_panelty = bool_masked_pos.mean(-1) # mask panelty to maximize the masked portion.
        mask_panelty = (mask * prob).sum(-1)  # 0s are the masked positions
        # mask_panelty = torch.exp(dist.log_prob(mask)) # [N,]
        # print(mask_panelty.shape)
        # print(reward.shape)
        reward = reward - mask_panelty # 0.2 is espilon in the formula


        # reward = reward + 0.5 * mask_panelty
        state = pixel_values
        return state, reward

        
    def train_one_batch(self, model, pixel_values: torch.Tensor, optimizer: torch.optim.Optimizer, num_steps=20, mini_batch_size=32, ppo_epochs=10):
        self.train() 
        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []
        entropy = 0
        state = pixel_values
        with torch.no_grad():
            for step in range(num_steps):
                dist, value = self.get_dist_critic(pixel_values)
                action = dist.sample()
                next_state, reward = self.get_action_reward(model, state, action, dist)

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward.unsqueeze(-1))
                if step == num_steps - 1:
                    masks.append(torch.zeros_like(reward).unsqueeze(-1))
                else:
                    masks.append(torch.ones_like(reward).unsqueeze(-1))
                
                states.append(state)
                actions.append(action)

                state = next_state 
            
            _, next_value = self.get_dist_critic(state)
            returns = self.compute_gae(next_value, rewards, masks, values)
            returns = torch.cat(returns)
            log_probs = torch.cat(log_probs)
            values    = torch.cat(values)
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            advantages = returns - values

        loss_dict = self.ppo_update(optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages)
        return loss_dict


    # def attribute_img(self, 
    #                   x, 
    #                   image_size=224, 
    #                   patch_size=16, 
    #                   baseline=None, 
    #                   seed=None):
    #     """
    #     Generate attribution heatmap for an input image.

    #     Args:
    #         x: An image tensor of shape [N, C, H, W], where H = W = image_size.
    #         image_size: The size of the input image (H = W = image_size).
    #         patch_size: The size of each patch. Can be used to calculate the number of tokens in each patch 
    #                     (image_size // patch_size) ** 2.
    #         baseline: The baseline tensor. If None, the baseline is set to the zero tensor.
    #         n_samples: The number of random masks to be generated.
    #         mask_prob: The probability of a token being masked, i.e., replaced by the baseline.
    #         seed: The seed value for random number generation.

    #     Returns:
    #         Attribution heatmap.
    #     """

    #     size = image_size // patch_size
    #     N, C, H, W = x.shape
    #     with torch.no_grad():
    #         predicted_class_selector = self.get_predicted_class_selector(x)
    #         outputs = self.forward(x, predicted_class_selector)
    #         # mask_list = outputs['mask_list']
    #         # probs_list = outputs['probs_list']
    #         probs = torch.sigmoid(outputs['sim']).reshape(N, size, size)
           
    #         # mask = torch.stack(mask_list, dim=1) # [N, n_samples, L]
    #         # probs = torch.stack(probs_list, dim=1) # [N, n_samples, 1]
    #         # weighted_mask = (mask * probs).sum(1) # [N, L]
    #         # weighted_mask = weighted_mask.reshape(N, size, size)
        
    #     return probs
    
    # def attribute_text(self, x):
    #     # TODO
    #     raise NotImplementedError("This function hasn't been developed.")
    