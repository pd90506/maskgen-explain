import torch
import torch.nn.functional as F
import math
from typing import Dict, Any
from maskgen.utils import idx_to_selector
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from maskgen.vision_models.vision_maskgen import MaskGeneratingModel


class PPOTrainer:
    def __init__(
        self,
        maskgen_model: MaskGeneratingModel,
        target_model: torch.nn.Module,
        config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.maskgen = maskgen_model.to(device)
        self.target_model = target_model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(self.maskgen.parameters(), lr=config['lr'])

    def get_action_reward(self, pixel_values: torch.Tensor, mask: torch.Tensor, 
                         label: torch.Tensor) -> tuple:
        """Calculate reward for the given action (mask).
        
        Args:
            pixel_values (torch.Tensor): Input image tensor of shape (N, C, H, W).
            mask (torch.Tensor): Generated mask tensor of shape (N, L) where L is number of patches.
            label (torch.Tensor): Model predicted label of shape (N, 1).
            
        Returns:
            tuple: A tuple containing:
                - pixel_values (torch.Tensor): Original input images
                - reward (torch.Tensor): Calculated reward tensor of shape (N,)
        """
        H, W = pixel_values.shape[-2:]

        pred_logits = self.target_model(pixel_values).logits # [N, num_classes]
        pred_prob = torch.softmax(pred_logits, -1).gather(1, label) # [N, 1]

        mask_size = int(math.sqrt(mask.shape[-1]))
        reshaped_mask = mask.reshape(-1, mask_size, mask_size).unsqueeze(1)
        upsampled_mask = F.interpolate(reshaped_mask, size=(H, W), mode='nearest')
        masked_input = pixel_values * upsampled_mask

        masked_pred_prob_all = torch.softmax(self.target_model(masked_input).logits, -1) # [N, num_classes]
        masked_true_prob = masked_pred_prob_all.gather(1, label) # [N, 1]
        
        reward = masked_true_prob * (torch.ones_like(mask).sum(-1, keepdim=True)) / (mask.sum(-1, keepdim=True) + 1)
        reward = torch.log(reward + 1e-6)
        
        return pixel_values, reward

    @torch.no_grad()
    def compute_gae(self, next_value, rewards, values):
        """Compute Generalized Advantage Estimation."""
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            gae = rewards[step] - values[step]
            returns.insert(0, gae + values[step])
        return returns

    def get_epsilon_greedy_action(self, dist, epsilon=0.05):
        """Implement epsilon-greedy action selection."""
        random_mask = (torch.rand_like(dist.probs) < epsilon)
        random_actions = torch.randint_like(dist.probs, low=0, high=2)
        sampled_actions = dist.sample()
        return torch.where(random_mask, random_actions, sampled_actions)

    def compute_losses(self, dist, value, mu_mean_prob, actions, returns, advantages, 
                      old_log_probs, logits):
        """Compute PPO losses."""
        new_log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
        ratio = (new_log_probs - old_log_probs).exp()
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config['clip_param'],
                           1.0 + self.config['clip_param']) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        critic_loss = (returns - value).pow(2).mean()

        entropy = dist.entropy().mean()
        
        log_logit = torch.log_softmax(logits, -1)
        kl_loss = F.kl_div(torch.log(mu_mean_prob), log_logit, 
                          reduction='batchmean', log_target=True)
        
        return actor_loss, critic_loss, entropy, kl_loss

    def collect_trajectories(self, pixel_values: torch.Tensor) -> dict:
        """Collect trajectories for PPO training.
        
        Args:
            pixel_values (torch.Tensor): Batch of input images
            
        Returns:
            dict: Contains collected trajectories with keys:
                - log_probs: List of action log probabilities
                - values: List of state values
                - states: List of states
                - actions: List of actions taken
                - rewards: List of rewards received
                - logits: List of original model logits
                - pseudo_labels: List of predicted labels
                - entropy: Total entropy of action distributions
        """
        # Initialize lists for PPO
        log_probs, values, states = [], [], []
        actions, rewards, masks = [], [], []
        entropy = 0
        logits, pseudo_labels = [], []

        # Get predicted labels from target model
        with torch.no_grad():
            pred_logits = self.target_model(pixel_values).logits
            pseudo_label = pred_logits.argmax(-1).unsqueeze(-1)
            logit = pred_logits.clone()

        state = pixel_values
        
        # Collect trajectories
        for step in range(self.config['num_steps']):
            with torch.no_grad():
                dist, value, _ = self.maskgen.get_dist_critic(pixel_values, pseudo_label)
                action = self.get_epsilon_greedy_action(dist, self.config['epsilon'])
                next_state, reward = self.get_action_reward(state, action, pseudo_label)

                log_prob = dist.log_prob(action).sum(-1, keepdim=True)
                entropy += dist.entropy().mean()
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            states.append(state.clone())
            actions.append(action.clone())
            logits.append(logit.clone())
            pseudo_labels.append(pseudo_label.clone())
            
            state = next_state

        return {
            'log_probs': log_probs,
            'values': values,
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'logits': logits,
            'pseudo_labels': pseudo_labels,
            'entropy': entropy
        }

    def train_epoch(self, dataloader, epoch, save_path):
        """Train for one epoch."""
        epoch_stats = {
            'loss': 0,
            'actor_loss': 0,
            'critic_loss': 0,
            'kl_loss': 0,
            'entropy': 0,
            'value': 0,
            'return': 0,
            'advantage': 0
        }
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            batch_stats = {k: 0 for k in epoch_stats}  # Initialize batch statistics
            pixel_values = batch['pixel_values'].to(self.device)
            
            # Collect trajectories
            trajectories = self.collect_trajectories(pixel_values)

            # Compute returns and advantages
            with torch.no_grad():
                _, next_value, _ = self.maskgen.get_dist_critic(pixel_values, trajectories['pseudo_labels'][-1])
                returns = self.compute_gae(next_value, trajectories['rewards'], trajectories['values'])
                
                returns = torch.cat(returns)
                log_probs = torch.cat(trajectories['log_probs'])
                values = torch.cat(trajectories['values'])
                states = torch.cat(trajectories['states'])
                actions = torch.cat(trajectories['actions'])
                logits = torch.cat(trajectories['logits'])
                pseudo_labels = torch.cat(trajectories['pseudo_labels'])
                advantages = returns - values

            # PPO update
            batch_size = states.size(0)
            for _ in range(self.config['ppo_epochs']):
                indices = torch.randperm(batch_size)
                for start in range(0, batch_size, self.config['mini_batch_size']):
                    end = start + self.config['mini_batch_size']
                    mb_indices = indices[start:end]
                    
                    dist, value, mu_mean_prob = self.maskgen.get_dist_critic(states[mb_indices], pseudo_labels[mb_indices])
                    
                    actor_loss, critic_loss, entropy, kl_loss = self.compute_losses(
                        dist, value, mu_mean_prob,
                        actions[mb_indices], returns[mb_indices],
                        advantages[mb_indices], log_probs[mb_indices],
                        logits[mb_indices]
                    )
                    
                    loss = (0.5 * critic_loss + 
                           self.config['l_actor'] * actor_loss - 
                           self.config['l_entropy'] * entropy +
                           self.config['l_kl'] * kl_loss)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Update batch stats
                    batch_stats['loss'] += loss.item()
                    batch_stats['actor_loss'] += actor_loss.item()
                    batch_stats['critic_loss'] += critic_loss.item()
                    batch_stats['kl_loss'] += kl_loss.item()
                    batch_stats['entropy'] += entropy.item()
            
            # Average batch stats
            num_updates = self.config['ppo_epochs'] * math.ceil(batch_size / self.config['mini_batch_size'])
            batch_stats = {k: v / num_updates for k, v in batch_stats.items()}

            # Track values, returns, and advantages before the current batch PPO updates
            batch_stats['value'] = values.mean().item()
            batch_stats['return'] = returns.mean().item()
            batch_stats['advantage'] = advantages.mean().item()
            
            # Log batch stats
            wandb.log({
                'batch': batch_idx,
                **batch_stats
            })
            
            # Update epoch stats (if needed)
            for k in epoch_stats:
                epoch_stats[k] += batch_stats[k]
            
            # Save checkpoint every 100 batches
            if (batch_idx + 1) % self.config['save_interval'] == 0:
                self.maskgen.save_model(f'{save_path}/maskgen_epoch_{epoch}_batch_{batch_idx}')
        
        # Average epoch stats (if needed)
        num_batches = len(dataloader)
        epoch_stats = {k: v / num_batches for k, v in epoch_stats.items()}
        
        return epoch_stats

    def train(self, train_dataloader):
        """Full training loop."""
        num_epochs = self.config['max_epochs']
        save_path = self.config['save_path']
        for epoch in range(num_epochs):
            self.maskgen.train()
            stats = self.train_epoch(train_dataloader, epoch, save_path)
            
            # Log stats
            wandb.log({
                'epoch': epoch,
                **stats
            })
            
            # Save checkpoint
            self.maskgen.save_model(f'{save_path}/maskgen_epoch_{epoch}')


def main():
    from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel, ViTConfig
    from maskgen.vision_models.vision_maskgen import MaskGeneratingModel, convert_to_peft
    from maskgen.utils import get_preprocess, collate_fn, load_imagenet

    # Training configuration
    config = {
        'batch_size': 32,
        'num_steps': 5,
        'mini_batch_size': 16,
        'ppo_epochs': 1,
        'epsilon': 0.0,
        'lr': 5e-5,
        'clip_param': 0.2,
        'l_kl': 0,
        'l_actor': 1.0,
        'l_entropy': 0.1,
        'gamma': 0.50,
        'tau': 0.95,
        'max_epochs': 5,
        'save_interval': 100,
        'save_path': './checkpoints'
    }

    # Initialize wandb
    wandb.init(project="maskgen", config=config)

    # Load models and processor
    pretrained_name = 'google/vit-base-patch16-224'
    vit_config = ViTConfig.from_pretrained(pretrained_name)
    processor = ViTImageProcessor.from_pretrained(pretrained_name)
    
    # Target model for explanation
    target_model = ViTForImageClassification.from_pretrained(pretrained_name)
    target_model.eval()

    # Create maskgen model
    base_model = ViTModel.from_pretrained(pretrained_name)
    peft_model = convert_to_peft(base_model)
    maskgen_model = MaskGeneratingModel(
        base_model=peft_model,
        hidden_size=vit_config.hidden_size,
        num_classes=vit_config.num_labels,
        freeze_base=False
    )

    # Data preprocessing
    dataset = load_imagenet(split='tiny')
    preprocess = get_preprocess(processor)

    dataset.set_transform(preprocess)
    
    # get dataloader
    train_dataloader = DataLoader(dataset, batch_size=config['batch_size'], 
                                  collate_fn=collate_fn, shuffle=True)

    # Initialize PPO model
    ppo_trainer = PPOTrainer(
        maskgen_model=maskgen_model,
        target_model=target_model,
        config=config
    )

    # Start training
    ppo_trainer.train(train_dataloader)

    # close wandb
    wandb.finish()

if __name__ == "__main__":
    main()