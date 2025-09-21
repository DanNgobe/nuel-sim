import random
from scipy.stats import beta
from .observation_model import ObservationModel

class BayesianMeanObservation(ObservationModel):
    def __init__(self, num_players, has_ghost=False, setup_shots=10):
        # Prevent re-initialization in singleton
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        self.num_players = num_players
        self.setup_shots = setup_shots
        self.has_ghost = has_ghost

        # Global prior Beta(α, β) for each player
        self.global_beliefs = {
            pid: {'alpha': 1.0, 'beta': 1.0, 'mean': 0.5} for pid in range(num_players + (1 if has_ghost else 0))
        }
        
        self._initialized = True

    @property
    def name(self) -> str:
        return "BayesianMeanObservation" + ("_with_abstention" if self.has_ghost else "") + f"_setup_{self.setup_shots}"


    def initialize(self, players):
        self.global_beliefs = {
            pid: {'alpha': 1.0, 'beta': 1.0, 'mean': 0.5} for pid in range(self.num_players + (1 if self.has_ghost else 0))
        }

        for _ in range(self.setup_shots):
            for player in players:
                if player.name == "Ghost":
                    self.global_beliefs[player.id]['mean'] = -1.0
                    continue

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
        
        others = [p for p in players if (p != player and p.name != "Ghost")]
        others.sort(key=lambda p: (self.global_beliefs[p.id]['mean'] if p.alive else 0.0), reverse=True)

        for p in others:
            obs.append(self.global_beliefs[p.id]['mean'] if p.alive else 0.0)
        return obs

    def get_observation_dim(self):
        return 1 + 1 + (self.num_players - 1)  # Self accuracy + mean + others' means

    def get_action_dim(self):
        return self.num_players - 1 + (1 if self.has_ghost else 0)

    def get_targets(self, player, players):
        others = [p for p in players if p != player]
        # Sort by bayesian mean belief, with ghosts at the end
        others.sort(key=lambda p: (
            (self.global_beliefs[p.id]['mean'] if p.alive else 0.0) if p.name != "Ghost" else -1.0
        ), reverse=True)
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
        """Reset bayesian beliefs to initial state"""
        self.global_beliefs = {
            pid: {'alpha': 1.0, 'beta': 1.0, 'mean': 0.5} for pid in range(self.num_players + (1 if self.has_ghost else 0))
        }
        if self.has_ghost:
            self.global_beliefs[self.num_players]['mean'] = -1.0  # Assuming ghost is the last player