import random
from scipy.stats import beta
from .observation_model import ObservationModel

class BayesianObservationModel(ObservationModel):
    def __init__(self, num_players, setup_shots = 10):
        self.num_players = num_players
        self.setup_shots = setup_shots
        # Global posterior Beta(α, β) for each player
        self.global_beliefs = {
            pid: {'alpha': 1.0, 'beta': 1.0, 'mean': 0.5} for pid in range(num_players)
        }

    def initialize(self, players):
        for _ in range(self.setup_shots):
            for player in players:
                # Simulate setup shots
                belief = self.global_beliefs[player.id]
                hit = random.random() < player.accuracy
                if hit:
                    belief['alpha'] += 1
                else:
                    belief['beta'] += 1
                belief['mean'] = belief['alpha'] / (belief['alpha'] + belief['beta'])

    def create_observation(self, player, players):
        obs = [player.accuracy, self.global_beliefs[player.id]['mean']]  # Self accuracy and mean belief
        
        others = [p for p in players if p != player]
        others.sort(key=lambda p: (self.global_beliefs[p.id]['mean'] if p.alive else 0.0), reverse=True)
       
        for p in others:
            obs.append(self.global_beliefs[p.id]['mean'] if p.alive else 0.0)
        return obs


    def get_observation_dim(self):
        return 1 + 1 + (self.num_players - 1)

    def get_action_dim(self):
        return self.num_players - 1  # can't target self

    def get_targets(self, player, players):
        others = [p for p in players if p != player]
        others.sort(key=lambda p: (self.global_beliefs[p.id]['mean'] if p.alive else 0.0), reverse=True)
        return others
    
    def update(self, shots, remaining_rounds=None):
        for shooter, target, hit in shots:
            belief = self.global_beliefs[shooter.id]
            if hit:
                belief['alpha'] += 1
            else:
                belief['beta'] += 1
            # Update mean
            belief['mean'] = belief['alpha'] / (belief['alpha'] + belief['beta'])

    
    def reset(self):
        self.global_beliefs = {pid: {'alpha': 1.0, 'beta': 1.0, 'mean': 0.5} for pid in range(self.num_players)}