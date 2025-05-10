from .observation_model import ObservationModel

class IndexAccuracyObservation(ObservationModel):
    def create_observation(self, player, players):
        index = players.index(player)
        index_one_hot = [1.0 if i == index else 0.0 for i in range(len(players))]
        obs = index_one_hot + [player.accuracy]
        for p in players:
            if p != player:
                obs.append(1.0 if p.alive else 0.0)
                obs.append(p.accuracy)
        return obs

    def get_observation_dim(self, num_players):
        return num_players + 1 + (num_players - 1) * 2

