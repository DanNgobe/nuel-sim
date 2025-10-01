# ppo_marl/agent.py
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from ppo_marl.network import ActorCriticLSTMNetwork
import ppo_marl.settings as settings
import numpy as np

class SharedPPOAgent:
    def __init__(self, observation_dim, action_dim, model_path=None):
        self.network = ActorCriticLSTMNetwork(observation_dim, action_dim, settings.HIDDEN_SIZE, settings.LSTM_LAYERS).to(settings.DEVICE)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=settings.LEARNING_RATE)
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.hidden_state = None

        if model_path:
            self.network.load_state_dict(torch.load(model_path, map_location=settings.DEVICE))
            
    def act(self, obs, explore=True):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=settings.DEVICE).unsqueeze(0)

        with torch.no_grad():
            action_logits, value, self.hidden_state = self.network(obs, self.hidden_state)
            
        if explore:
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item(), value.item()
        else:
            action = action_logits.argmax(dim=1)
            return action.item(), None, None

    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values[-1]
            else:
                next_value = values[i + 1]
                
            delta = rewards[i] + settings.GAMMA * next_value * (1 - dones[i]) - values[i]
            gae = delta + settings.GAMMA * settings.GAE_LAMBDA * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            
        return torch.tensor(advantages, dtype=torch.float32, device=settings.DEVICE)

    def update(self, memory):
        if len(memory) < settings.BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones, old_log_probs, values = memory.get_batch()
        
        # Compute next values for GAE
        with torch.no_grad():
            _, next_values, _ = self.network(next_states)
            next_values = next_values.squeeze()
        
        # Compute advantages and returns
        advantages = self.compute_gae(rewards, values, next_values, dones)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        for _ in range(settings.PPO_EPOCHS):
            # Get current policy and value predictions
            action_logits, current_values, _ = self.network(states)
            current_values = current_values.squeeze()
            
            # Calculate policy loss
            dist = Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Importance sampling ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - settings.CLIP_EPSILON, 1 + settings.CLIP_EPSILON) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(current_values, returns)
            
            # Total loss
            total_loss = policy_loss + settings.VALUE_COEFF * value_loss - settings.ENTROPY_COEFF * entropy
            
            # Update network
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.optimizer.step()

    def reset_hidden_state(self):
        """Reset LSTM hidden state"""
        self.hidden_state = None
    
    def save_model(self, path):
        """Save the network"""
        torch.save(self.network.state_dict(), path)
        
    def load_model(self, path):
        """Load the network"""
        self.network.load_state_dict(torch.load(path, map_location=settings.DEVICE))