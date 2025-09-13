import os
from typing import List, Optional, TYPE_CHECKING
from core.strategies import BaseStrategy
from ppo_marl.agent import SharedPPOAgent

if TYPE_CHECKING:
    from core.player import Player

class PPOStrategy(BaseStrategy):
    """PPO-based strategy that implements BaseStrategy interface (Singleton)"""

    _instance = None

    def __new__(cls, observation_model, model_path=None, explore=True):
        if cls._instance is None:
            cls._instance = super(PPOStrategy, cls).__new__(cls)
        return cls._instance

    def __init__(self, observation_model, model_path=None, explore=True):
        # Prevent reinitialization in singleton
        if hasattr(self, "_initialized") and self._initialized:
            return
        super().__init__("ppo_strategy")
        self.observation_model = observation_model
        self.explore = explore

        # Create the agent
        observation_dim = observation_model.get_observation_dim()
        action_dim = observation_model.get_action_dim()

        if not model_path or not os.path.exists(model_path):
            model_path = None
            print("Model path does not exist, starting with untrained agent.")

        self.agent = SharedPPOAgent(observation_dim, action_dim, model_path=model_path)

        # Set to evaluation mode if not exploring
        if not explore:
            self.agent.network.eval()

        self._initialized = True

    def choose_target(self, me: "Player", players: List["Player"], observation=None) -> tuple[Optional["Player"], Optional[int]]:
        if observation is None:
            observation = self.observation_model.create_observation(me, players)

        # Get action from agent
        if self.explore:
            action, log_prob, value = self.agent.act(observation, explore=True)
            # Store additional info for training
            self._last_log_prob = log_prob
            self._last_value = value
        else:
            action, _, _ = self.agent.act(observation, explore=False)

        # Get valid targets
        targets = self.observation_model.get_targets(me, players)

        # Return the chosen target
        if action < len(targets):
            return targets[action], action
        return None, None

    def get_agent(self):
        """Get the underlying PPO agent"""
        return self.agent
    
    def get_last_training_info(self):
        """Get last log_prob and value for training"""
        return getattr(self, '_last_log_prob', None), getattr(self, '_last_value', None)