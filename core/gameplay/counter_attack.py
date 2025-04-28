import random
from .base import GamePlay

# If target survives, it counterattacks the attacker
class CounterAttackGamePlay(GamePlay):
    def choose_shooters(self, alive_players, last_shooter):
        if not alive_players:
            return []
        attacker = random.choice(alive_players)
        return [attacker]

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
