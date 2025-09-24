# ppo_marl/settings.py
import torch

GAMMA = 0.99
LEARNING_RATE = 3e-4
CLIP_EPSILON = 0.2
VALUE_COEFF = 0.5
ENTROPY_COEFF = 0.01
BATCH_SIZE = 64
MEMORY_SIZE = 2048
HIDDEN_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PPO_EPOCHS = 4
GAE_LAMBDA = 0.95

def get_model_path(num_players, gameplay_name, observation_name):
    """Generate model path based on game configuration"""
    return f"ppo_marl/models/{num_players}/{gameplay_name}/{observation_name}.pth"