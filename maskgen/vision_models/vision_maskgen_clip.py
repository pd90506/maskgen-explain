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
import logging


class MaskGeneratingModel(nn.Module):
    def __init__(self, base_model:nn.Module, hidden_size, num_classes, config, freeze_base=True):
        super().__init__()
        self.bce_loss = nn.BCELoss(reduction='none')
        self.base_model = base_model
        self.config = config

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
        
        self.num_classes = num_classes

        # Freeze the base_model
        if freeze_base:
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

        output = self.base_model(pixel_values)

        hidden_states = output['last_hidden_state'][:,1:,:]
        # hidden_states = hidden_states + label_emb.unsqueeze(1)
        # mu_logits = self.actor(hidden_states).squeeze(-1) # [N, L]
        mu_logits = self.actor(hidden_states) # [N, L, num_classes]
        # we need softmax probability, instead of logits, to learn multiple classes in a single shot.
        mu_prob = torch.softmax(mu_logits, -1) # [N, L, num_classes]   
        # mu_prob = torch.sigmoid(mu_logits) # [N, L, num_classes]
        # mu_logits = F.normalize(mu_logits, p=1, dim=-1)
        labels_expanded = labels.unsqueeze(1).expand(-1, mu_logits.shape[1]) # [N, L]
        labels_expanded = labels_expanded.unsqueeze(-1) # [N, L, 1]
        # mu_logits = torch.gather(mu_logits, -1, labels_expanded).squeeze(-1) # [N, L]
        mu_prob = torch.gather(mu_prob, -1, labels_expanded).squeeze(-1) # [N, L]
        # print(mu_logits[0])
        dist = Bernoulli(probs=mu_prob)
        # TODO 改成logit？
        # dist = Bernoulli(logits=mu_logits)

        pooled_output = output['last_hidden_state'][:,0,:] # [N, d]
        # pooled_output = pooled_output + label_emb
        value = self.critic(pooled_output) # [N, num_classes]
        value = value.gather(1, labels.unsqueeze(-1)) # [N,]

        return dist, value, mu_logits


    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage, logits, true_labels):
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
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :], logits[rand_ids, :], true_labels[rand_ids, :]
            
            
    def ppo_update(self, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, logits, true_labels, clip_param=0.2):
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
            for state, action, old_log_probs, return_, advantage, logit, true_label in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages, logits, true_labels):

                dist, value, mu_logits = self.get_dist_critic(state, labels=true_label)

                log_logit = torch.log_softmax(logit, -1)
                mu_softmax = torch.softmax(mu_logits, -1)
                mu_logits_sum_softmax = mu_softmax.mean(1)
                # mu_logits_sum_softmax = mu_logits_sum_softmax / mu_logits_sum_softmax.sum(-1, keepdim=True)
                log_mu_logits = torch.log(mu_logits_sum_softmax)
                kl_div = F.kl_div(log_mu_logits, log_logit, reduction='batchmean', log_target=True)

                # logit_expanded = logit.unsqueeze(1).expand_as(mu_logits)
                # mu_logits_softmax = torch.log_softmax(mu_logits, -1)
                # logit_expand_softmax = torch.softmax(logit_expanded, -1)
                # kl_div = F.kl_div(mu_logits_softmax, logit_expand_softmax, reduction='batchmean')
                # kl_div = kl_div / mu_logits_softmax.shape[1]
                
                # mu_prob = torch.sigmoid(mu_logits)
                
                # mu_prob_mean = mu_prob.mean(1) # [N, num_classes]
                # log_mu_prob_mean = torch.log(mu_prob_mean / mu_prob_mean.sum(-1, keepdim=True))
                # log_mu_prob_mean = torch.log(mu_prob_mean)
                
                # kl_div = F.kl_div(log_mu_prob_mean, log_logit, reduction='batchmean', log_target=True)
                # mu_logits_sum_log_softmax = torch.log(mu_logits_sum_softmax)
                # logging.debug(f"Shapes - mu_logits: {mu_logits_sum_log_softmax.shape}")
                # logging.debug(f"mu_logits: {mu_logits_sum_log_softmax[0]}")
                
                # true_dist, true_value = self.get_dist_critic(state, labels=true_label)
                # value = value - true_value

                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action).sum(-1, keepdim=True)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                # PPO step
                actor_loss  = - torch.min(surr1, surr2).mean()
                # learn the value function based on the estimated return
                critic_loss = (return_ - value).pow(2).mean() 

                loss = 0.5 * critic_loss + actor_loss - 0.0001 * entropy + self.config['l_kl'] * kl_div

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
                "kl_div": kl_div.item()}
    
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
        # total_rewards = [sum(rewards[:i+1]) for i in range(len(rewards))]
        gae = 0
        returns = []
        for idx, step in enumerate(reversed(range(len(rewards)))):
            gae = rewards[step] - values[step]
            returns.insert(0, gae + values[step])
        return returns

    
    @torch.no_grad()
    def get_action_reward(self, model, pixel_values, mask, label, true_label):
        """ 
        action : mask
        state : pixel_values
        """
        H, W = pixel_values.shape[-2:]

        pred_logits = model(pixel_values).logits # [N, num_classes]
        # pred = pred_logits.argmax(-1) # [N,]
        # TODO: change pred to label
        true_selector = idx_to_selector(true_label, self.num_classes) # [N, num_classes]

        pred_prob = torch.softmax(pred_logits, -1) # [N, num_classes]
        pred_prob = (pred_prob * true_selector).sum(-1) # [N,]

        # bool_masked_pos = 1.0 - mask
        mask_size = int(math.sqrt(mask.shape[-1]))
        reshaped_mask = mask.reshape(-1, mask_size, mask_size).unsqueeze(1) # [N, 1, size, size]
        upsampled_mask = F.interpolate(reshaped_mask, size=(H, W), mode='nearest') # [N, 1, H, W]
        masked_input = pixel_values * upsampled_mask # [N, C, H, W]

        masked_pred_prob_all = torch.softmax(model(masked_input).logits, -1) # [N, num_classes]

        masked_true_prob = (masked_pred_prob_all * true_selector).sum(-1)

        reward = masked_true_prob

        reward = reward * (torch.ones_like(mask).sum(-1)) / (mask.sum(-1) + 1) # [N,]

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
        logits = [] 
        true_labels = []

        state = pixel_values
        with torch.no_grad():
            pred_logits = model(pixel_values).logits # [N, num_classes]
            true_label = pred_logits.argmax(-1).unsqueeze(-1) # [N, 1]

        # TODO: replace the labels with random dummy labels
        # label = torch.randint(0, self.num_classes, (pixel_values.shape[0], 1)).to(pixel_values.device)
        # replace the labels with the second most probable label
        # label = torch.topk(pred_logits, 2, dim=-1)[1][:, 1].unsqueeze(-1)
        logit = pred_logits.clone()

        def get_epsilon_greedy_action(dist, epsilon=0.05):
            # Create a random mask with shape same as dist.probs
            random_mask = (torch.rand_like(dist.probs) < epsilon)

            # Get both random actions and sampled actions
            random_actions = torch.randint_like(dist.probs, low=0, high=2)
            sampled_actions = dist.sample()

            # Combine them using the mask
            action = torch.where(random_mask, random_actions, sampled_actions)
            return action

    
        with torch.no_grad():
            for step in range(num_steps):
                dist, value, _ = self.get_dist_critic(pixel_values, labels=true_label)

                action = get_epsilon_greedy_action(dist, epsilon=self.config['epsilon'])
                
                next_state, reward = self.get_action_reward(model, state, action, logit, true_label.squeeze(-1))

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
                logits.append(logit.clone())
                true_labels.append(true_label.clone())

                state = next_state 
            
            _, next_value, _ = self.get_dist_critic(state, true_label)
            
            returns = self.compute_gae(next_value, rewards, masks, values)
            returns = torch.cat(returns)
            log_probs = torch.cat(log_probs)
            values    = torch.cat(values)
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            logits    = torch.cat(logits)
            true_labels = torch.cat(true_labels)

            advantages = returns - values

        loss_dict = self.ppo_update(optimizer=optimizer, 
                                    ppo_epochs=ppo_epochs, 
                                    mini_batch_size=mini_batch_size,
                                    states=states, 
                                    actions=actions, 
                                    log_probs=log_probs, 
                                    returns=returns, 
                                    advantages=advantages, 
                                    logits=logits, 
                                    true_labels=true_labels, 
                                    clip_param=self.config['clip_param'])
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
            dist, value, _ = self.get_dist_critic(pixel_values=pixel_values, labels=labels)
            sim_probs = dist.probs
            heatmap = sim_probs.reshape(N, size, size)
        
        return heatmap
    
    # def attribute_text(self, x):
    #     # TODO
    #     raise NotImplementedError("This function hasn't been developed.")
    