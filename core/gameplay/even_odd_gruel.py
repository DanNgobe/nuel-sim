import random
from .base import GamePlay

class EvenOddGruelGamePlay(GamePlay):
    def choose_shooters(self, alive_players):
        return alive_players[:]  # all alive players participate

    def conduct_shots(self, shooters, players):
        shots = []

        for shooter in shooters:
            possible_targets = self.get_valid_targets(shooter, players)
            if possible_targets:
                target = random.choice(possible_targets)
            else:
                target = None

            hit = shooter.shoot(target)
            shots.append((shooter, target, hit))

        return shots

    def process_shots(self, shots):
        dead_players = set()
        for shooter, target, hit in shots:
            if hit and target and target.alive:
                dead_players.add(target)

        for player in dead_players:
            player.alive = False

    def get_valid_targets(self, shooter, players):
        # Find alive targets of opposite parity
        shooter_index = players.index(shooter)
        shooter_is_even = shooter_index % 2 == 0

        valid_targets = []
        for i, player in enumerate(players):
            if player.alive and player != shooter:
                target_is_even = i % 2 == 0
                if shooter_is_even != target_is_even:
                    valid_targets.append(player)

        return valid_targets

    def is_over(self, players, alive_players):
        # check if only one player is alive or all alive players are of the same parity
        if len(alive_players) <= 1:
            return True
        # Check if all alive players are of the same parity
        first_parity = players.index(alive_players[0]) % 2
        for player in alive_players[1:]:
            if players.index(player) % 2 != first_parity:
                return False
        return True
    
