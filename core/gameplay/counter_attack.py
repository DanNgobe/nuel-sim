import random
from .base import GamePlay

# If target survives, it counterattacks the attacker
class CounterAttackRandomGamePlay(GamePlay):
    def choose_shooters(self, eligible_players):
        shooter = random.choice(eligible_players)
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