# config.py

from core.gameplay import SequentialGamePlay, RandomGamePlay, SimultaneousGamePlay, CounterAttackGamePlay, EvenOddGruelGamePlay
GAME_PLAY = SimultaneousGamePlay()

from core.strategies import target_strongest, target_weakest, target_stronger, target_stronger_or_strongest, target_nearest, target_random
DEFAULT_STRATEGY = target_random

# Game Settings
NUM_PLAYERS = 3

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
