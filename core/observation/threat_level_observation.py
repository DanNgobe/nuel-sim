from .observation_model import ObservationModel

class ThreatLevelObservation(ObservationModel):
    def __init__(self, num_players, has_ghost=False):
        self.num_players = num_players
        self.has_ghost = has_ghost

    @property
    def name(self) -> str:
        return "ThreatLevelObservation" + ("_with_abstention" if self.has_ghost else "")

    def create_observation(self, player, players):
        obs = [player.accuracy]  # Self accuracy

        others = [p for p in players if (p != player and p.name != "Ghost")]
        # Sort by threat level: alive * accuracy
        others.sort(key=lambda p: (p.accuracy if p.alive else 0.0), reverse=True)

        for p in others:
            obs.append(p.accuracy if p.alive else 0.0)
        return obs

    def get_observation_dim(self):
        return 1 + (self.num_players - 1)

    def get_action_dim(self):
        return self.num_players - 1 + (1 if self.has_ghost else 0) 

    def get_targets(self, player, players):
        others = [p for p in players if p != player]
        # Targets must match the sorted order in the observation
        others.sort(key=lambda p: (p.accuracy if p.alive else 0.0), reverse=True)
        return others
