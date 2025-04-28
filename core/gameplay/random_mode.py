import random
from .base import GamePlay

#Random alive player is attacker, only one shot per turn
class RandomGamePlay(GamePlay):
    def choose_shooters(self, alive_players, last_shooter):
        if not alive_players:
            return []
        return [random.choice(alive_players)]
