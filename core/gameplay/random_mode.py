import random
from .base import GamePlay

#Random alive player is attacker, only one shot per turn
class RandomGamePlay(GamePlay):
    def __init__(self):
        super().__init__()
        self.already_shot = set()

    def choose_shooters(self, alive_players, last_shooter=None):
        eligible_players = [p for p in alive_players if p not in self.already_shot]
        if not eligible_players:
            self.already_shot.clear()
            eligible_players = alive_players

        shooter = random.choice(eligible_players)
        self.already_shot.add(shooter)
        return [shooter]
    
    def is_over(self, players):
        over = super().is_over(players)
        if over:
            self.already_shot.clear()
        return over
