from abc import ABC, abstractmethod

class GamePlay(ABC):
    """Abstract base class for game play strategies."""
    
    @abstractmethod
    def choose_shooters(self, eligible_players):
        pass

    def is_over(self, players, alive_players):
        return len(alive_players) <= 1

    def conduct_shots(self, shooters, players):
        shots = []
        for shooter in shooters:
            target = shooter.choose_target(players)
            hit = shooter.shoot(target)
            shots.append((shooter, target, hit))
        return shots

    def process_shots(self, shots):
        for shooter, target, hit in shots:
            if hit and target and target.alive:
                target.alive = False
