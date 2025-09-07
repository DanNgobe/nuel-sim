from .observation_model import ObservationModel

class SimpleObservation(ObservationModel):
    def __init__(self, num_players, has_ghost=False):
        self.num_players = num_players
        self.has_ghost = has_ghost

    @property
    def name(self) -> str:
        return "SimpleObservation" + ("_with_abstention" if self.has_ghost else "")

    def create_observation(self, player, players):
        obs = [player.accuracy]  # Self accuracy
        
        # Create a consistent ordering by player ID
        other_players = [p for p in players if (p != player and p.name != "Ghost")]
        other_players.sort(key=lambda p: p.id)  # Sort by player ID for consistency
        
        for p in other_players:
            obs.append(p.accuracy if p.alive else 0.0)
    
        return obs

    def get_observation_dim(self):
        return 1 + (self.num_players - 1)

    def get_action_dim(self):
        return self.num_players - 1 + (1 if self.has_ghost else 0)

    def get_targets(self, player, players):
        # Return other players sorted by ID for consistent action mapping
        other_players = [p for p in players if p != player]
        other_players.sort(key=lambda p: p.id)
        return other_players