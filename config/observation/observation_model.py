from abc import ABC, abstractmethod

class ObservationModel(ABC):
    @abstractmethod
    def create_observation(self, player, players): pass

    @abstractmethod
    def get_observation_dim(self, num_players): pass

    @abstractmethod
    def get_action_dim(self, num_players): pass
