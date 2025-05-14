from abc import ABC, abstractmethod

class ObservationModel(ABC):
    @abstractmethod
    def create_observation(self, player, players): pass

    @abstractmethod
    def get_observation_dim(self): pass

    @abstractmethod
    def get_action_dim(self): pass

    @abstractmethod
    def get_targets(self, player, players): pass
