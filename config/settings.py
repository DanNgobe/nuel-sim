# settings.py
get_model_path = lambda num_players, game_play, observation_model: f"marl/models/{num_players}_{game_play.__class__.__name__}_{observation_model.__class__.__name__}.pth"

from .observation import  IndexAccuracyObservation, SortedAccuracyObservation
from marl.utils import create_agent, agent_based_strategy
from core.gameplay import SequentialGamePlay, RandomGamePlay, SimultaneousGamePlay, CounterAttackRandomGamePlay, EvenOddGruelGamePlay

GAME_PLAY = SimultaneousGamePlay()

from core.strategies import target_strongest, target_weakest, target_stronger, target_stronger_or_strongest, target_nearest, target_random

# Game Settings
NUM_PLAYERS = 5
OBSERVATION_MODEL = SortedAccuracyObservation(num_players=NUM_PLAYERS)
MARL_AGENT = create_agent(OBSERVATION_MODEL, model_path=get_model_path(NUM_PLAYERS, GAME_PLAY, OBSERVATION_MODEL), is_evaluation=True)
DEFAULT_STRATEGY = agent_based_strategy(OBSERVATION_MODEL, MARL_AGENT, explore=False)

# Player Settings
# INITIAL_HEALTH = 1.0
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
