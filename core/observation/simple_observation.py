from .observation_model import ObservationModel

class SimpleObservation(ObservationModel):
    """
    Simple observation model that sorts targets by player ID.
    Provides basic player information without complex belief tracking.
    """
    
    def __init__(self, num_players, has_ghost=True):
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        self.num_players = num_players
        self.has_ghost = has_ghost
        self._initialized = True

    @property
    def name(self) -> str:
        return "SimpleObservation" + ("_with_ghost" if self.has_ghost else "")
    
    def create_observation(self, player, players):
        obs = [player.accuracy]  # Self accuracy
        others = [p for p in players if p != player and p.name != "Ghost"]
        others.sort(key=lambda p: p.id)
        obs.extend(p.accuracy if p.alive else 0.0 for p in others)
        return obs

    def get_observation_dim(self):
        return 1 + (self.num_players - 1)

    def get_action_dim(self):
        return (self.num_players - 1) + (1 if self.has_ghost else 0)

    def get_targets(self, player, players):
        others = [p for p in players if p != player]
        others.sort(key=lambda p: p.id)        
        return others