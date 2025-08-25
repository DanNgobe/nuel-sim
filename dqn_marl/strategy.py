import os
from typing import List, Optional, TYPE_CHECKING
from core.strategies import BaseStrategy
from dqn_marl.agent import SharedAgent

if TYPE_CHECKING:
    from core.player import Player

class DQNStrategy(BaseStrategy):
    """DQN-based strategy that implements BaseStrategy interface"""
    
    def __init__(self, observation_model, model_path=None, explore=True):
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
    
    def choose_target(self, me: "Player", players: List["Player"], observation=None) -> tuple[Optional["Player"], Optional[int]]:
        if observation is None:
            observation = self.observation_model.create_observation(me, players)

        # Get action from agent
        action = self.agent.act(observation, self.explore)

        # Get valid targets
        targets = self.observation_model.get_targets(me, players)
        
        # Return the chosen target
        if action < len(targets):
            return targets[action], action
        return None, None

    def get_agent(self):
        """Get the underlying DQN agent"""
        return self.agent
