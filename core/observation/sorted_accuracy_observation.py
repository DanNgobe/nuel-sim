from .observation_model import ObservationModel

class SortedAccuracyObservation(ObservationModel):
    def __init__(self, num_players):
        self.num_players = num_players

    def create_observation(self, player, players):
        obs = [player.accuracy]  # Include self accuracy first

        others = [p for p in players if p != player]
        # Sort other players by their accuracy (descending = most dangerous first)
        others.sort(key=lambda p: p.accuracy, reverse=True)

        for p in others:
            obs.append(1.0 if p.alive else 0.0)
            obs.append(p.accuracy)

        return obs

    def get_observation_dim(self):
        return 1 + 2 * (self.num_players - 1)  # self.accuracy + (alive + accuracy) per other player

    def get_action_dim(self):
        return self.num_players - 1

    def get_targets(self, player, players):
        # Return targets in the same sorted order
        others = [p for p in players if p != player]
        others.sort(key=lambda p: p.accuracy, reverse=True)
        return others
