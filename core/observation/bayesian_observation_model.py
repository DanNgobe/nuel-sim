import random
import math
from scipy.stats import beta
from .observation_model import ObservationModel

def beta_variance(alpha, beta):
    numerator = alpha * beta
    denominator = (alpha + beta) ** 2 * (alpha + beta + 1)
    return numerator / denominator

class BayesianObservationModel(ObservationModel):
    def __init__(self, num_players, has_ghost=False, setup_shots=10, ucb_scale=1.0):
        self.num_players = num_players
        self.setup_shots = setup_shots
        self.has_ghost = has_ghost
        self.ucb_scale = ucb_scale

        self.global_beliefs = {
            pid: {
                'alpha': 1.0,
                'beta': 1.0,
                'mean': 0.5,
                'ucb': 0.5  # initially same as mean
            } for pid in range(num_players + (1 if has_ghost else 0))
        }

    @property
    def name(self) -> str:
        return "BayesianObservationModel" + ("_with_abstention" if self.has_ghost else "") + f"_setup_{self.setup_shots}_ucb_{self.ucb_scale}"

    def _update_belief_stats(self, belief):
        alpha, beta = belief['alpha'], belief['beta']
        mean = alpha / (alpha + beta)
        var = beta_variance(alpha, beta)
        ucb = mean + self.ucb_scale * math.sqrt(var)
        belief['mean'] = mean
        belief['ucb'] = ucb

    def initialize(self, players):
        for _ in range(self.setup_shots):
            for player in players:
                belief = self.global_beliefs[player.id]
                if player.name == "Ghost":
                    belief['mean'] = -1.0
                    belief['ucb'] = -1.0
                    continue

                hit = random.random() < player.accuracy
                if hit:
                    belief['alpha'] += 1
                else:
                    belief['beta'] += 1
                self._update_belief_stats(belief)

    def create_observation(self, player, players):
        obs = [player.accuracy, self.global_beliefs[player.id]['ucb']]  # Self accuracy + UCB belief

        others = [p for p in players if (p != player and p.name != "Ghost")]
        others.sort(key=lambda p: (self.global_beliefs[p.id]['ucb'] if p.alive else 0.0), reverse=True)

        for p in others:
            obs.append(self.global_beliefs[p.id]['ucb'] if p.alive else 0.0)
        return obs

    def get_observation_dim(self):
        return 1 + 1 + (self.num_players - 1)  # Self accuracy + own UCB + others' UCBs

    def get_action_dim(self):
        return self.num_players - 1 + (1 if self.has_ghost else 0)

    def get_targets(self, player, players):
        others = [p for p in players if p != player]
        others.sort(key=lambda p: (self.global_beliefs[p.id]['ucb'] if p.alive else 0.0), reverse=True)
        return others

    def update(self, shots, remaining_rounds=None):
        for shooter, target, hit in shots:
            belief = self.global_beliefs[shooter.id]
            if hit:
                belief['alpha'] += 1
            else:
                belief['beta'] += 1
            self._update_belief_stats(belief)

    def reset(self):
        self.global_beliefs = {
            pid: {
                'alpha': 1.0,
                'beta': 1.0,
                'mean': 0.5,
                'ucb': 0.5
            } for pid in range(self.num_players + (1 if self.has_ghost else 0))
        }
        if self.has_ghost:
            self.global_beliefs[self.num_players]['mean'] = -1.0
            self.global_beliefs[self.num_players]['ucb'] = -1.0
