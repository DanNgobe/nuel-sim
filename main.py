import math
import random
from core import Player
from core import Game
from visual import run_game_visual
import config

def create_players(n):
    center_x, center_y, radius = config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2, 200
    players = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        accuracy = config.ASSIGNED_DEFAULT_ACCURACIES[i] if i < len(config.ASSIGNED_DEFAULT_ACCURACIES) else random.uniform(*config.MARKSMANSHIP_RANGE)
        strategy = config.ASSIGNED_DEFAULT_STRATEGIES[i] if i < len(config.ASSIGNED_DEFAULT_STRATEGIES) else config.DEFAULT_STRATEGY
        players.append(Player(id=i, name=f"P{i+1}", accuracy=accuracy, x=x, y=y, strategy=strategy))
    return players

if __name__ == "__main__":
    players = create_players(config.NUM_PLAYERS)
    game = Game(players, gameplay=config.GAME_PLAY, observation_model=config.OBSERVATION_MODEL, max_rounds=config.NUM_ROUNDS)
    run_game_visual(game)
