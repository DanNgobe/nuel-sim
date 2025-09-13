# ppo_marl/memory.py
import torch
import numpy as np
from ppo_marl.settings import DEVICE

class PPOMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.clear()

    def push(self, state, action, reward, next_state, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def get_batch(self):
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(self.actions, dtype=torch.long, device=DEVICE)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=DEVICE)
        next_states = torch.tensor(np.array(self.next_states), dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=DEVICE)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=DEVICE)
        values = torch.tensor(self.values, dtype=torch.float32, device=DEVICE)
        
        return states, actions, rewards, next_states, dones, log_probs, values

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def __len__(self):
        return len(self.states)