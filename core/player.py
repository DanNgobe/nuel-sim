import random

class Player:
    def __init__(self, name, accuracy, x=0, y=0):
        self.name = name
        self.accuracy = accuracy
        self.x = x
        self.y = y
        self.alive = True

    def choose_target(self, players):
        """Override or extend this for smarter targeting"""
        alive = [p for p in players if p != self and p.alive]
        return random.choice(alive) if alive else None

    def shoot(self, target):
        if not self.alive or not target or not target.alive:
            return False
        hit = random.random() < self.accuracy
        if hit:
            target.alive = False
        return hit
