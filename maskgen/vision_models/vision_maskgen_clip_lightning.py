import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import ViTModel
from torch.distributions import Bernoulli
import numpy as np 
from maskgen.utils import idx_to_selector
import torch.nn.functional as F


class MaskGeneratingModel(pl.LightningModule):
    def __init__(self, base_model, pred_model, hidden_size, num_classes):
        super().__init__()
        self.base_model = base_model
        self.pred_model = pred_model
        # Freeze pred_model 的所有参数
        for param in self.pred_model.parameters():
            param.requires_grad = False

        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
        self.num_classes = num_classes

    def get_dist_critic(self, pixel_values, labels):
        labels = labels.squeeze(-1)
        selector = idx_to_selector(labels, self.num_classes) # [N, num_classes]
        output = self.base_model(pixel_values)

        hidden_states = output['last_hidden_state'][:, 1:, :]
        mu_logits = self.actor(hidden_states) # [N, L, num_classes]
        mu_logits = (mu_logits * selector.unsqueeze(1)).sum(-1) # [N, L]
        dist = Bernoulli(logits=mu_logits)

        pooled_output = output['last_hidden_state'][:, 0, :] # [N, d]
        value = self.critic(pooled_output) # [N, num_classes]
        value = (value * selector).sum(-1, keepdim=True) # [N, 1]

        return dist, value

    @torch.no_grad()
    def get_action_reward(self, model, pixel_values, mask):
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
        reward = reward * (torch.ones_like(mask).sum(-1)) / (mask.sum(-1) + 1e-5) # [N,]

        state = pixel_values
        return state, reward

    def training_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        # Training logic 
        loss_dict = self.train_one_batch(self.pred_model, pixel_values, self.optimizers())
        self.log("train_loss", loss_dict["loss"])

        # return loss_dict
        return None

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)

    def train_one_batch(self, model, pixel_values, optimizer, num_steps=5, mini_batch_size=256, ppo_epochs=2):
        self.train() 
        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []
        entropy = 0
        labels = []

        # 确保 pixel_values 在正确的设备上
        pixel_values = pixel_values.to(next(self.parameters()).device)
        state = pixel_values
        with torch.no_grad():
            pred_logits = model(pixel_values).logits # [N, num_classes]
            label = pred_logits.argmax(-1).unsqueeze(-1) # [N, 1]
    
        with torch.no_grad():
            for step in range(num_steps):
                dist, value = self.get_dist_critic(pixel_values, label)
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
            # returns = torch.cat(returns)
            returns = returns[0].repeat(len(rewards), 1)
            log_probs = torch.cat(log_probs)
            values    = torch.cat(values)
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            labels = torch.cat(labels)
            # returns = returns[0]
            # log_probs = log_probs[0]
            # values    = values[0]
            # states    = states[0]
            # actions   = actions[0]
            # labels = labels[0]

            advantages = returns - values
        


        loss_dict = self.ppo_update(optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, labels)
        return loss_dict

    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        sum_gae = 0
        returns = []
        for idx, step in enumerate(reversed(range(len(rewards)))):
            delta = rewards[step] - values[step]
            sum_gae = delta + sum_gae
            gae = sum_gae / (idx + 1)
            returns.insert(0, gae + values[step])
        return returns

    def ppo_update(self, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, labels, clip_param=0.2):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage, label in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages, labels):
                dist, value = self.get_dist_critic(state, label)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action).sum(-1, keepdim=True)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.0001 * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return {"loss": loss.item(),
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "returns": return_.mean().item(),
                "entropy": entropy.item()}

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantages, labels):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantages[rand_ids, :], labels[rand_ids, :]

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

# 示例训练
# 创建模型
# model = MaskGeneratingModel(hidden_size=768, num_classes=1000)

# # 使用 Trainer 进行训练
# trainer = pl.Trainer(gpus=torch.cuda.device_count(), max_epochs=10)
# trainer.fit(model, train_dataloader)
