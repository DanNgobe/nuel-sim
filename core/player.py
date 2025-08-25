import random

class Player:
    def __init__(self, id, name, accuracy, x=0, y=0, strategy=None, alive=True):
        self.id = id
        self.name = name
        self.accuracy = accuracy
        self.x = x
        self.y = y
        self.alive = alive
        
        # Import strategy here to avoid circular import
        if strategy is None:
            from .strategies import TargetRandom
            strategy = TargetRandom()
        self.strategy = strategy

    def choose_target(self, players):
        return self.strategy.choose_target(self, players)

    def shoot(self, target):
        if not self.alive or not target:
            raise ValueError("Player is not alive or target is None")
        return random.random() < self.accuracy