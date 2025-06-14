from abc import ABC, abstractmethod
from core.player import Player

class NullObservationModel:
    @property
    def name(self) -> str:
        """Name of the null observation model."""
        return "NullObservationModel"

    def initialize(self, players: list[Player]):
        """Null observation model does nothing."""
        pass
    def update(self, shots: list[tuple], remaining_rounds:  int | None = None):
        """Null observation model does nothing."""
        pass
    def reset(self):
        """Null observation model does nothing."""
        pass

class ObservationModel(NullObservationModel, ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the observation model."""
        pass

    @abstractmethod
    def create_observation(self, player: Player, players: list[Player]): pass

    @abstractmethod
    def get_observation_dim(self): pass

    @abstractmethod
    def get_action_dim(self): pass

    @abstractmethod
    def get_targets(self, player: Player, players: list[Player]): pass