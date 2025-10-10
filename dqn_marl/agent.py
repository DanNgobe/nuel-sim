# dqn_marl/agent.py
import random
import torch
import torch.nn.functional as F
from dqn_marl.network import PolicyNetwork
import dqn_marl.settings as settings
import numpy as np

class SharedAgent:
    """
    Double DQN (DDQN) agent implementation for improved stability.
    Uses separate policy and target networks to decouple action selection 
    and value estimation, reducing overestimation bias.
    """
    def __init__(self, observation_dim, action_dim, model_path=None):
        # Double DQN: Policy network for action selection and target network for value estimation
        self.policy_net = PolicyNetwork(observation_dim, action_dim, settings.HIDDEN_SIZE).to(settings.DEVICE)
        self.target_net = PolicyNetwork(observation_dim, action_dim, settings.HIDDEN_SIZE).to(settings.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=settings.LEARNING_RATE)
        self.epsilon = settings.EPSILON_START
        self.action_dim = action_dim
        self.observation_dim = observation_dim

        if model_path:
            self.policy_net.load_state_dict(torch.load(model_path, map_location=settings.DEVICE))
            
    def act(self, obs, explore=True):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=settings.DEVICE).unsqueeze(0)

        if explore and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                q_values = self.policy_net(obs)
            return q_values.argmax(dim=1).item()

    def update(self, replay_buffer):
        if len(replay_buffer) < settings.BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones, idxs, is_weights = replay_buffer.sample(settings.BATCH_SIZE)
        is_weights = torch.tensor(is_weights, dtype=torch.float32, device=settings.DEVICE)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=settings.DEVICE)
        actions = torch.tensor(actions, dtype=torch.long, device=settings.DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=settings.DEVICE)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=settings.DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32, device=settings.DEVICE)

        # Current Q-values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: Decouple action selection and evaluation to reduce overestimation bias
        with torch.no_grad():
            # Select action with max Q-value from policy net
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            # Evaluate using target net
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        
        expected_q = rewards + settings.GAMMA * next_q_values * (1 - dones)
        
        # Calculate TD errors for priority updates
        td_errors = (q_values - expected_q).detach().cpu().numpy()

        # Weighted loss using importance sampling weights
        loss = (is_weights * F.mse_loss(q_values, expected_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update priorities
        replay_buffer.update_priorities(idxs, td_errors)

    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self, episode, total_episodes=25000):
        # Linear decay: gradually decrease epsilon from START to END over total_episodes
        decay_rate = (settings.EPSILON_START - settings.EPSILON_END) / (total_episodes * 0.25)
        self.epsilon = max(settings.EPSILON_END, settings.EPSILON_START - decay_rate * episode)

    def save_model(self, path):
        """Save the policy network"""
        torch.save(self.policy_net.state_dict(), path)
        
    def load_model(self, path):
        """Load the policy network"""
        self.policy_net.load_state_dict(torch.load(path, map_location=settings.DEVICE))
