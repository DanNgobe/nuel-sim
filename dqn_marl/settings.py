# dqn_marl/settings.py
import torch

GAMMA = 0.99 # Discount factor for future rewards (closer to 1 makes agent more future-oriented)
LEARNING_RATE = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.98
BATCH_SIZE = 64 # Larger batches smooth gradient updates; smaller ones update faster.
MEMORY_SIZE = 10000 # Increase to 50,000+ for more complex environments.
HIDDEN_SIZE = 128 # Don't overfit on small environments by going too large.
TARGET_UPDATE_FREQUENCY = 1000 # Update target network every N steps for DDQN stability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prioritized Experience Replay settings
PRIORITY_ALPHA = 0.6  # How much prioritization is used (0 = uniform, 1 = full prioritization)
PRIORITY_BETA = 0.4   # Importance sampling correction (0 = no correction, 1 = full correction)
PRIORITY_BETA_INCREMENT = 0.001  # Beta annealing rate

# Model path configuration
def get_model_path(num_players, gameplay_name, observation_name):
    """Generate model path based on game configuration"""
    return f"dqn_marl/models/{num_players}/{gameplay_name}/{observation_name}.pth"
