import math
import random
from core import Player
from core import Game
from visual import run_game_visual

def create_players(n):
    center_x, center_y, radius = 400, 300, 200
    players = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        players.append(Player(f"P{i+1}", accuracy=random.uniform(0.5, 0.9), x=x, y=y))
    return players

if __name__ == "__main__":
    players = create_players(10)
    game = Game(players, mode="sequential")  # or "simultaneous" or "random"
    run_game_visual(game)
