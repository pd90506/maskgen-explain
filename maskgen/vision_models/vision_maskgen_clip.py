import torch 
import torch.nn as nn
from maskgen.utils import idx_to_selector
import torch.nn.functional as F
import numpy as np
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

        self.label_embedding = nn.Embedding(num_classes, hidden_size)

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
        
        self.num_classes = num_classes

        # Freeze the base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

    
    def get_dist_critic(self, pixel_values, labels):
        """
        Calculates the distribution of the mask and the critic value given the pixel values.
        Args:
            pixel_values (torch.Tensor) [N, C, H, W]: Tensor containing the pixel values.
            labels (torch.Tensor) [N, 1]: Tensor containing the labels.
        Returns:
            dist (torch.distributions.Bernoulli): Bernoulli distribution of mask.
            value (torch.Tensor): The critic value.
        """
        labels = labels.view(-1)
        label_emb = self.label_embedding(labels) # [N, d]
        # print("label_emb", label_emb.shape)
        output = self.base_model(pixel_values)

        hidden_states = output['last_hidden_state'][:,1:,:]
        hidden_states = hidden_states + label_emb.unsqueeze(1)
        mu_logits = self.actor(hidden_states).squeeze(-1) # [N, L]
        dist = Bernoulli(logits=mu_logits)

        pooled_output = output['last_hidden_state'][:,0,:] # [N, d]
        pooled_output = pooled_output + label_emb
        value = self.critic(pooled_output) # [N, 1]

        return dist, value


    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage, labels):
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
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :], labels[rand_ids, :]
            
            
    def ppo_update(self, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, labels, clip_param=0.2):
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
            for state, action, old_log_probs, return_, advantage, label in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages, labels):

                dist, value = self.get_dist_critic(state, labels=label)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action).sum(-1, keepdim=True)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                # PPO step
                actor_loss  = - torch.min(surr1, surr2).mean()
                # learn the value function based on the estimated return
                critic_loss = (return_ - value).pow(2).mean()

                mask_loss = dist.logits.mean(-1).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy + 0.1 * mask_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Print the losses after one ppo update
        return  {"loss": loss.item(), 
                 "actor_loss": actor_loss.item(), 
                 "critic_loss": critic_loss.item(),
                 "returns": return_.mean().item(),
                 'entropy': entropy.item(),
                 "value": value.mean().item(),
                 "mask": mask_loss.item()}
    
    @torch.no_grad()
    def compute_gae(self, next_value, rewards, masks, values, gamma=0.50, tau=0.95):
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
        sum_gae=0
        returns = []
        mean_reward = sum(rewards) / len(rewards)
        for idx, step in enumerate(reversed(range(len(rewards)))):
            # delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            # gae = rewards[step] - gamma * mean_reward - (1 - gamma) * values[step]
            
            # discounted_value =  (1 - gamma) * step * values[step]
            # discounted_value_diff = (1 - gamma) * values[step]
            # delta = rewards[step] - discounted_value_diff
            # gae  = delta + tau * gae
            gae = rewards[step] - values[step]

 
            # gae = delta + gamma * tau * masks[step] * gae
            # sum_gae = delta + sum_gae
            # gae = sum_gae / (idx + 1)


            returns.insert(0, gae + values[step])
            # print(delta.mean())
        return returns

    
    @torch.no_grad()
    def get_action_reward(self, model, pixel_values, mask):
        """ 
        action : mask
        state : pixel_values
        """
        H, W = pixel_values.shape[-2:]

        pred_logits = model(pixel_values).logits # [N, num_classes]
        pred = pred_logits.argmax(-1) # [N,]
        selector = idx_to_selector(pred, self.num_classes) # [N, num_classes]

        pred_prob = torch.softmax(pred_logits, -1) # [N, num_classes]
        pred_prob = (pred_prob * selector).sum(-1) # [N,]

        # bool_masked_pos = 1.0 - mask
        mask_size = int(math.sqrt(mask.shape[-1]))
        reshaped_mask = mask.reshape(-1, mask_size, mask_size).unsqueeze(1) # [N, 1, size, size]
        upsampled_mask = F.interpolate(reshaped_mask, size=(H, W), mode='nearest') # [N, 1, H, W]
        masked_input = pixel_values * upsampled_mask # [N, C, H, W]

        # masked_pred_logits = model(pixel_values, bool_masked_pos=bool_masked_pos).logits # [N, num_classes]
        masked_pred_prob = torch.softmax(model(masked_input).logits, -1) # [N, num_classes]
        masked_pred_prob = (masked_pred_prob * selector).sum(-1) # [N,]

        reward = torch.exp(torch.log(masked_pred_prob) - torch.log(pred_prob)) # [N,]
        reward = reward * (torch.ones_like(mask).sum(-1)) / (mask.sum(-1) + 1) # [N,]
        # print("reward", reward[0])

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
        labels = [] 

        # # 确保 pixel_values 在正确的设备上
        # pixel_values = pixel_values.to(next(self.parameters()).device)
        state = pixel_values
        # with torch.no_grad():
        #     pred_logits = model(pixel_values).logits # [N, num_classes]
        #     label = pred_logits.argmax(-1).unsqueeze(-1) # [N, 1]

        # replace the labels with random dummy labels
        label = torch.randint(0, self.num_classes, (pixel_values.shape[0], 1)).to(pixel_values.device)
    
        with torch.no_grad():
            for step in range(num_steps):
                dist, value = self.get_dist_critic(pixel_values, labels=label)
                action = dist.sample()
                next_state, reward = self.get_action_reward(model, state, action)

                log_prob = dist.log_prob(action).sum(-1, keepdim=True)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward.unsqueeze(-1))
                if step == num_steps - 1:
                    masks.append(torch.zeros_like(reward).unsqueeze(-1))
                else:
                    masks.append(torch.ones_like(reward).unsqueeze(-1))
                
                states.append(state.clone())
                actions.append(action.clone())
                labels.append(label.clone())


                state = next_state 
            
            _, next_value = self.get_dist_critic(state, label)
            
            returns = self.compute_gae(next_value, rewards, masks, values)
            # returns = self.compute_gae_static_state(next_value, rewards, masks, values)
            returns = torch.cat(returns)
            # returns = returns[0].repeat(len(rewards), 1)
            log_probs = torch.cat(log_probs)
            values    = torch.cat(values)
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            labels    = torch.cat(labels)
            # returns = returns[0]
            # log_probs = log_probs[0]
            # values    = values[0]
            # states    = states[0]
            # actions   = actions[0]

            advantages = returns - values
        
        # print("states", states.shape)
        # print("actions", actions.shape)
        # print("log_probs", log_probs.shape)
        # print("returns", returns.shape)
        # print("advantages", advantages.shape)
        # print("labels", labels.shape)


        loss_dict = self.ppo_update(optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, labels)
        return loss_dict


    def attribute_img(self, 
                      pixel_values, 
                      labels,
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
        N, C, H, W = pixel_values.shape
        with torch.no_grad():
            dist, value = self.get_dist_critic(pixel_values=pixel_values, labels=labels)
            sim_logits = dist.logits
            heatmap = torch.sigmoid(sim_logits).reshape(N, size, size)
        
        return heatmap
    
    # def attribute_text(self, x):
    #     # TODO
    #     raise NotImplementedError("This function hasn't been developed.")
    