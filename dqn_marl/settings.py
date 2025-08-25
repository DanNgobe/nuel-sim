# dqn_marl/settings.py
import torch

GAMMA = 0.99 # Discount factor for future rewards (closer to 1 makes agent more future-oriented)
LEARNING_RATE = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.2
EPSILON_DECAY = 0.98
BATCH_SIZE = 64 # Larger batches smooth gradient updates; smaller ones update faster.
MEMORY_SIZE = 10000 # Increase to 50,000+ for more complex environments.
HIDDEN_SIZE = 128 # Don't overfit on small environments by going too large.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model path configuration
def get_model_path(num_players, gameplay_name, observation_name):
    """Generate model path based on game configuration"""
    return f"dqn_marl/models/{num_players}_{gameplay_name}_{observation_name}.pth"
