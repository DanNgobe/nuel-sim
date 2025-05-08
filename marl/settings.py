# marl/settings.py
GAMMA = 0.99 # Discount factor for future rewards (closer to 1 makes agent more future-oriented)
LEARNING_RATE = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.98
BATCH_SIZE = 64
MEMORY_SIZE = 50000
TARGET_UPDATE_FREQ = 10  # How often to update target network
HIDDEN_SIZE = 128
DEVICE = "cpu"  # or "cuda"
