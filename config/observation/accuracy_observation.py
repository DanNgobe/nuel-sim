from .observation_model import ObservationModel

class AccuracyObservation(ObservationModel):

    def __init__(self, num_players):
        self.num_players = num_players


    def create_observation(self, player, players):
        obs = [player.accuracy]
        for p in players:
            if p != player:
                obs.append(1.0 if p.alive else 0.0)
                obs.append(p.accuracy)
        return obs

    def get_observation_dim(self, num_players):
        return 1 + (num_players - 1) * 2

    def get_action_dim(self, num_players):
        return num_players - 1
