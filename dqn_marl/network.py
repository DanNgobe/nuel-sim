import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_heads=None):
        super().__init__()
        if num_heads is None:
            num_heads = min(4, input_dim) if input_dim % 4 == 0 else min(2, input_dim) if input_dim % 2 == 0 else 1
        self.attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        self.proj_to_hidden = nn.Linear(input_dim, hidden_size)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size//2, hidden_size)
        )
        
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_out)
        
        x = self.proj_to_hidden(x)
        
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)
        
        x = x[:, -1, :]
        return self.output_proj(x)