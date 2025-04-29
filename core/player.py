import random
import config

class Player:
    def __init__(self, name, accuracy, x=0, y=0, strategy=None):
        self.name = name
        self.accuracy = accuracy
        self.x = x
        self.y = y
        self.alive = True
        self.strategy = strategy or config.DEFAULT_STRATEGY

    def default_strategy(self, me, players):
        """Default target selection: random alive enemy."""
        alive = [p for p in players if p != me and p.alive]
        return random.choice(alive) if alive else None

    def choose_target(self, players):
        return self.strategy(self, players)

    def shoot(self, target):
        if not self.alive or not target or not target.alive:
            return False
        return random.random() < self.accuracy