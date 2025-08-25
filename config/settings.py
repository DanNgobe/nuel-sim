# settings.py
from core.observation import ThreatLevelObservation, BayesianObservationModel, TurnAwareThreatObservation
from core.gameplay import SequentialGamePlay, RandomGamePlay, SimultaneousGamePlay, CounterAttackRandomGamePlay, EvenOddGruelGamePlay
from core.strategies import TargetStrongest, TargetWeaker, TargetStronger, TargetRandom, TargetNearest

# Game Settings
NUM_PLAYERS = 3
NUM_ROUNDS = None
RUN_MODE = "single"  # "visualize", or "single"

# Gameplay
GAME_PLAY = SimultaneousGamePlay()
HAS_GHOST = False

# Observation Model
OBSERVATION_MODEL = ThreatLevelObservation(NUM_PLAYERS, has_ghost=HAS_GHOST)
MODEL_PATH = f"marl/models/{NUM_PLAYERS}_{GAME_PLAY.__class__.__name__}_{OBSERVATION_MODEL.name}.pth"

# Player Settings
MARKSMANSHIP_RANGE = (0.3, 0.9)

TargetStrongestStrategy = TargetStrongest()
# Strategies (now using class instances)
ASSIGNED_STRATEGIES = [
    TargetStrongestStrategy,
    TargetStrongestStrategy,
    TargetStrongestStrategy
]

ASSIGNED_ACCURACIES = []

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