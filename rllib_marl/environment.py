from core.game_manager import GameManager
from core.gameplay import SimultaneousGamePlay
from core.observation import ThreatLevelObservation

def NuelMultiAgentEnv(config):
    """Factory function to create GameManager with RLlib config"""
    return GameManager(
        num_players=config.get("num_players", 3),
        gameplay=config.get("gameplay", SimultaneousGamePlay()),
        observation_model=config.get("observation_model", ThreatLevelObservation(3)),
        max_rounds=config.get("max_rounds", None),
        marksmanship_range=config.get("marksmanship_range", (0.3, 0.9)),
        strategies=[],  # Empty for RL training
        assigned_accuracies=config.get("assigned_accuracies", []),
        has_ghost=config.get("has_ghost", False),
    )