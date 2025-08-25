# marl/agent.py
import random
import torch
import torch.nn.functional as F
from ddqn_marl.network import PolicyNetwork
import ddqn_marl.settings as settings

class SharedAgent:
    def __init__(self, observation_dim, action_dim, model_path=None):
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

        states, actions, rewards, next_states, dones = replay_buffer.sample(settings.BATCH_SIZE)

        states = torch.tensor(states, dtype=torch.float32, device=settings.DEVICE)
        actions = torch.tensor(actions, dtype=torch.long, device=settings.DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=settings.DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=settings.DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32, device=settings.DEVICE)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Avoids overestimation by decoupling action selection and evaluation
        # Select action with max Q-value from policy net
        # Evaluate using target net
        next_actions = self.policy_net(next_states).argmax(1, keepdim=True) 
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1) # or next_q_values = self.target_net(next_states).max(1)[0]
        
        expected_q = rewards + settings.GAMMA * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, expected_q)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self, episode, total_episodes=25000):
        # Linear decay: gradually decrease epsilon from START to END over total_episodes
        decay_rate = (settings.EPSILON_START - settings.EPSILON_END) / (total_episodes * 0.25)
        self.epsilon = max(settings.EPSILON_END, settings.EPSILON_START - decay_rate * episode)
