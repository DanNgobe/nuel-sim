import random
from .base import GamePlay

#Random alive player is attacker, only one shot per turn
class RandomGamePlay(GamePlay):
    def choose_shooters(self, eligible_players):
        shooter = random.choice(eligible_players)
        return [shooter]
