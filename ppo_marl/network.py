import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticLSTMNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, lstm_layers=1, num_heads=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        
        # Multi-head attention on raw input
        if num_heads is None:
            num_heads = min(4, input_dim) if input_dim % 4 == 0 else min(2, input_dim) if input_dim % 2 == 0 else 1
        self.attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        
        # Project to hidden size after attention
        self.proj_to_hidden = nn.Linear(input_dim, hidden_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, lstm_layers, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Multi-head attention on raw features
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_out)
        
        # Project to hidden dimension
        x = self.proj_to_hidden(x)
        
        # LSTM processing
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_features = self.layer_norm2(lstm_out[:, -1, :])  # Take last timestep
        
        action_logits = self.actor(lstm_features)
        value = self.critic(lstm_features)
        return action_logits, value, hidden
    
    def get_action_probs(self, x, hidden=None):
        action_logits, _, hidden = self.forward(x, hidden)
        return F.softmax(action_logits, dim=-1), hidden
    
    def get_value(self, x, hidden=None):
        _, value, hidden = self.forward(x, hidden)
        return value, hidden
    
    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=device))