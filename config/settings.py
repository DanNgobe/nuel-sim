# settings.py

from core.observation import  ThreatLevelObservation, BayesianObservationModel, TurnAwareThreatObservation
from marl.utils import create_agent, agent_based_strategy
from core.gameplay import SequentialGamePlay, RandomGamePlay, SimultaneousGamePlay, CounterAttackRandomGamePlay, EvenOddGruelGamePlay
from core.strategies import target_strongest, target_weakest, target_stronger, target_stronger_or_strongest, target_nearest, target_random


GAME_PLAY = SequentialGamePlay()
HAS_GHOST = False

# Game Settings
NUM_PLAYERS = 3
NUM_ROUNDS = None

OBSERVATION_MODEL = ThreatLevelObservation(NUM_PLAYERS, has_ghost=HAS_GHOST)
MODEL_PATH = f"marl/models/{NUM_PLAYERS}_{GAME_PLAY.__class__.__name__}_{OBSERVATION_MODEL.name}.pth"
MARL_AGENT = create_agent(OBSERVATION_MODEL, model_path=MODEL_PATH, is_evaluation=True)
DEFAULT_STRATEGY = agent_based_strategy(OBSERVATION_MODEL, MARL_AGENT, explore=False)

# Player Settings
MARKSMANSHIP_RANGE = (0.3, 0.9)

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


# ASSIGNED DEFAULTS (mixed strategies)
ASSIGNED_DEFAULT_STRATEGIES = []
ASSIGNED_DEFAULT_ACCURACIES = []
