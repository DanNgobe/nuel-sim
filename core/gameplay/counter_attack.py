import random
from .base import GamePlay

# If target survives, it counterattacks the attacker
class CounterAttackRandomGamePlay(GamePlay):
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

    def conduct_shots(self, shooters, players):
        shots = []
        if not shooters:
            return shots

        attacker = shooters[0]
        target = attacker.choose_target(players)
        hit = attacker.shoot(target)
        shots.append((attacker, target, hit))

        # Counterattack if target survived
        if target and target.alive:
            counter_hit = target.shoot(attacker)
            shots.append((target, attacker, counter_hit))

        return shots
    
    def is_over(self, players):
        over = super().is_over(players)
        if over:
            self.already_shot.clear()
        return over
