# settings.py
# Configuration with NO imports from core modules to prevent circular dependencies
# Use string identifiers that will be resolved by factories at runtime

# Game Settings
NUM_PLAYERS = 5
NUM_ROUNDS = 20  # None for infinite
RUN_MODE = "visualize"  # "visualize", or "single"

# Gameplay Configuration
# Available options: "SequentialGamePlay", "RandomGamePlay", "SimultaneousGamePlay"
GAME_PLAY_TYPE = "SimultaneousGamePlay"
HAS_GHOST = True

# Observation Model Configuration
# Available options: "SortedObservation", "BayesianMeanObservation", "BayesianAbstentionObservation", "TurnAwareThreatObservation", "SimpleObservation", "NoInfoObservation"
OBSERVATION_MODEL_TYPE = "BayesianAbstentionObservation"
OBSERVATION_MODEL_PARAMS = {
    "num_players": NUM_PLAYERS,
    "has_ghost": HAS_GHOST,
    "setup_shots": 0,
}

# Player Settings
MARKSMANSHIP_RANGE = (0.3, 0.9)

# Strategy Configuration (string identifiers)
# Available options: "TargetStrongest", "TargetWeaker", "TargetStronger", "TargetRandom", "TargetNearest", "RLlibStrategy", "DQNStrategy", "PPOStrategy"
ASSIGNED_STRATEGY_TYPES = ["PPOStrategy"] * NUM_PLAYERS
ASSIGNED_ACCURACIES = []

# RLlib Strategy Configuration
import os
_ALGORITHM = "ppo"  # "ppo" or "dqn"
RLLIB_CHECKPOINT_PATH = os.path.abspath(f"rllib_marl/checkpoints/{_ALGORITHM}/{NUM_PLAYERS}_players/{GAME_PLAY_TYPE}_{OBSERVATION_MODEL_TYPE}")

# Pygame Visual Settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PLAYER_RADIUS = 30
FPS = 1
ARROW_SIZE = 15

# Colors (R, G, B)
COLOR_BACKGROUND = (30, 30, 30)
COLOR_PLAYER_ALIVE = (0, 150, 255)
COLOR_PLAYER_DEAD = (100, 100, 100)
COLOR_HIT = (0, 255, 0)
COLOR_MISS = (255, 0, 0)