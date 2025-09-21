import os
from typing import List, Optional, TYPE_CHECKING
from core.strategies import BaseStrategy
from dqn_marl.agent import SharedAgent

if TYPE_CHECKING:
    from core.player import Player

class DQNStrategy(BaseStrategy):
    """DQN-based strategy that implements BaseStrategy interface (Singleton)"""

    _instance = None

    def __new__(cls, observation_model, model_path=None, explore=True):
        if cls._instance is None:
            cls._instance = super(DQNStrategy, cls).__new__(cls)
        return cls._instance

    def __init__(self, observation_model, model_path=None, explore=True):
        # Prevent reinitialization in singleton
        if hasattr(self, "_initialized") and self._initialized:
            return
        super().__init__("dqn_strategy")
        self.observation_model = observation_model
        self.explore = explore

        # Create the agent
        observation_dim = observation_model.get_observation_dim()
        action_dim = observation_model.get_action_dim()

        if not model_path or not os.path.exists(model_path):
            model_path = None
            print("Model path does not exist, starting with untrained agent.")

        self.agent = SharedAgent(observation_dim, action_dim, model_path=model_path)

        # Set to evaluation mode if not exploring
        if not explore:
            self.agent.policy_net.eval()

        self._initialized = True

    def choose_target(self, me: "Player", players: List["Player"], observation=None) -> tuple[Optional["Player"], Optional[int]]:
        if observation is None:
            observation = self.observation_model.create_observation(me, players)

        # Get action from agent
        action = self.agent.act(observation, self.explore)

        # Use the new robust action-to-target mapping
        target = self.observation_model.get_target_from_action(me, players, action)
        
        # Return the chosen target (None if action was invalid)
        return target, action if target is not None else None

    def get_agent(self):
        """Get the underlying DQN agent"""
        return self.agent
