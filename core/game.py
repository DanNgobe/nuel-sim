import random

class Game:
    def __init__(self, players, mode="sequential"):
        self.players = players
        self.history = []
        self.turn = 0
        self.mode = mode  # "sequential", "simultaneous", or "random"

    def get_alive_players(self):
        return [p for p in self.players if p.alive]

    def is_over(self):
        return len(self.get_alive_players()) <= 1

    def run_turn(self):
        alive = self.get_alive_players()
        if self.mode == "simultaneous":
            shots = []
            for shooter in alive:
                target = shooter.choose_target(alive)
                hit = shooter.shoot(target)
                shots.append((shooter, target, hit))
            self.history.extend(shots)

        else:  # sequential or random
            if self.mode == "random":
                shooter = random.choice(alive)
            else:
                shooter = alive[self.turn % len(alive)]
            target = shooter.choose_target(alive)
            hit = shooter.shoot(target)
            self.history.append((shooter, target, hit))
            self.turn += 1
