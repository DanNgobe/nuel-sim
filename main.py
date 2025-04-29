import math
import random
from core import Player
from core import Game
from visual import run_game_visual
from marl.strategy import agent_strategy
import config

def create_players(n):
    center_x, center_y, radius = config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2, 200
    players = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        accuracy = random.uniform(*config.MARKSMANSHIP_RANGE)
        players.append(Player(f"P{i+1}", accuracy=accuracy, x=x, y=y, strategy=agent_strategy()))
    return players

if __name__ == "__main__":
    players = create_players(config.NUM_PLAYERS)
    game = Game(players, gameplay=config.GAME_PLAY)
    run_game_visual(game)
