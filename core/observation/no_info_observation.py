from .observation_model import ObservationModel

class NoInfoObservation(ObservationModel):
    """Observation model that provides no information - all zeros."""
    
    def __init__(self, num_players, has_ghost=False, setup_shots=0):
        self.num_players = num_players
        self.has_ghost = has_ghost

    @property
    def name(self) -> str:
        return "NoInfoObservation"

    def initialize(self, players):
        pass

    def create_observation(self, player, players):
        return [0.0] * self.get_observation_dim()

    def get_observation_dim(self):
        return self.num_players - 1

    def get_action_dim(self):
        return (self.num_players - 1) + (1 if self.has_ghost else 0)

    def get_targets(self, player, players):
        others = [p for p in players if p != player]
        others.sort(key=lambda p: p.id)
        return others
    
    def update(self, shots, remaining_rounds=None):
        pass

    def reset(self):
        pass