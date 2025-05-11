from .observation_model import ObservationModel

class IndexAccuracyObservation(ObservationModel):
    def __init__(self, num_players, game_play):
        is_simultaneous = 'Simultaneous' in game_play.__class__.__name__
        is_random = 'Random' in game_play.__class__.__name__
        self.include_index = not (is_simultaneous or is_random)
        self.num_players = num_players

    def create_observation(self, player, players):
        obs = []
        if self.include_index:
            index = players.index(player)
            index_one_hot = [1.0 if i == index else 0.0 for i in range(len(players))]
            obs.extend(index_one_hot)
        obs.append(player.accuracy)
        for p in players:
            if p != player:
                obs.append(1.0 if p.alive else 0.0)
                obs.append(p.accuracy)
        return obs

    def get_observation_dim(self):
        index_dim = self.num_players if self.include_index else 0
        return index_dim + 1 + 2 * (self.num_players - 1)
    
    def get_action_dim(self):
        return self.num_players - 1

