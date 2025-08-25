import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),  # Better than BatchNorm for RL
            nn.ReLU(),
            nn.Dropout(0.1),  # Prevent overfitting
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)