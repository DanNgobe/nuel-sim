import random

class Game:
    def __init__(self, players, mode="sequential"):
        self.players = players
        self.history = [] # [[(shooter, target, hit), ...], ...]
        self.last_shooter = None
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
            self.history.append(shots)

        else:  # sequential or random
            if self.mode == "random":
                shooter = random.choice(alive)
            else:  # sequential using last_shooter
                shooter = None
                start_index = 0

                # If we have a last shooter, start from the next index
                if self.last_shooter in self.players:
                    start_index = (self.players.index(self.last_shooter) + 1) % len(self.players)

                for i in range(len(self.players)):
                    idx = (start_index + i) % len(self.players)
                    candidate = self.players[idx]
                    if candidate.alive:
                        shooter = candidate
                        break

            if shooter:
                target = shooter.choose_target(alive)
                hit = shooter.shoot(target)
                self.history.append([(shooter, target, hit)])
                self.last_shooter = shooter
