from abc import ABC, abstractmethod

class GamePlay(ABC):
    """Abstract base class for game play strategies."""
    
    @abstractmethod
    def choose_shooters(self, eligible_players):
        pass

    def is_over(self, players, alive_players):
        return len(alive_players) <= 1

    def process_shots(self, shots):
        for shooter, target, hit in shots:
            if hit and target and target.alive:
                target.alive = False
