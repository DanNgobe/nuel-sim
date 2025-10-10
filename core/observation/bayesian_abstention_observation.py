import random
from scipy.stats import beta
from .observation_model import ObservationModel

class BayesianAbstentionObservation(ObservationModel):
    """Bayesian observation model with abstention support."""
    
    def __init__(self, num_players, has_ghost=True, setup_shots=10):
        if hasattr(self, "_initialized") and self._initialized:
            return
            
        self.num_players = num_players
        self.setup_shots = setup_shots
        self.has_ghost = has_ghost

        self.global_beliefs = {
            pid: {'alpha': 1.0, 'beta': 1.0, 'mean': 0.5} 
            for pid in range(num_players)
        }
        
        self.shot_counts = {
            pid: 0 for pid in range(num_players)
        }
        
        self._initialized = True

    @property
    def name(self) -> str:
        return "BayesianAbstentionObservation" + ("_with_abstention" if self.has_ghost else "") + f"_setup_{self.setup_shots}"

    def initialize(self, players):
        self.global_beliefs = {
            pid: {'alpha': 1.0, 'beta': 1.0, 'mean': 0.5} 
            for pid in range(self.num_players)
        }
        self.shot_counts = {
            pid: 0 for pid in range(self.num_players)
        }

        for _ in range(self.setup_shots):
            for player in players:
                if player.name == "Ghost":
                    continue

                belief = self.global_beliefs[player.id]
                hit = random.random() < player.accuracy
                if hit:
                    belief['alpha'] += 1
                else:
                    belief['beta'] += 1
                belief['mean'] = belief['alpha'] / (belief['alpha'] + belief['beta'])
                self.shot_counts[player.id] += 1
    
    def create_observation(self, player, players):
        obs = [
            player.accuracy, 
            self.global_beliefs[player.id]['mean']
        ]
        
        others = [p for p in players if (p != player and p.name != "Ghost")]
        others.sort(key=lambda p: (self.global_beliefs[p.id]['mean'] if p.alive else 0.0), reverse=True)

        for p in others:
            obs.append(self.global_beliefs[p.id]['mean'] if p.alive else 0.0)
        
        for p in others:
            obs.append(self.shot_counts[p.id] if p.alive else 0.0)
        
        return obs

    def get_observation_dim(self):
        return 2 + 2 * (self.num_players - 1)

    def get_action_dim(self):
        return (self.num_players - 1) + (1 if self.has_ghost else 0)

    def get_targets(self, player, players):
        others = [p for p in players if p != player]
        others.sort(key=lambda p: (
            (self.global_beliefs[p.id]['mean'] if p.alive else 0.0) if p.name != "Ghost" else -1.0
        ), reverse=True)
        return others
    
    def update(self, shots, remaining_rounds=None):
        for shooter, target, hit in shots:
            if shooter.name == "Ghost" or (target and target.name == "Ghost"):
                continue # If you abstain, don't update beliefs

            if shooter.id in self.shot_counts:
                self.shot_counts[shooter.id] += 1
                
            if shooter.id in self.global_beliefs:
                belief = self.global_beliefs[shooter.id]
                if hit:
                    belief['alpha'] += 1
                else:
                    belief['beta'] += 1
                belief['mean'] = belief['alpha'] / (belief['alpha'] + belief['beta'])

    def reset(self):
        self.global_beliefs = {
            pid: {'alpha': 1.0, 'beta': 1.0, 'mean': 0.5} 
            for pid in range(self.num_players)
        }
        self.shot_counts = {
            pid: 0 for pid in range(self.num_players)
        }