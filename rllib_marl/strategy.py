# Import warning suppression first
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import suppress_warnings
suppress_warnings.suppress_ray_warnings()

import numpy as np
import torch
from typing import List, Optional, TYPE_CHECKING
from core.strategies import BaseStrategy
from ray.rllib.algorithms import Algorithm

if TYPE_CHECKING:
    from core.player import Player

class RLlibStrategy(BaseStrategy):
    """RLlib-based strategy that implements BaseStrategy interface using the new RLModule API (Singleton)"""

    _instance = None

    def __new__(cls, checkpoint_path, observation_model=None, policy_id="shared_policy"):
        if cls._instance is None:
            cls._instance = super(RLlibStrategy, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, checkpoint_path, observation_model=None, policy_id="shared_policy"):
        if getattr(self, "_initialized", False):
            return
        super().__init__("rllib_strategy")
        self.observation_model = observation_model
        self.policy_id = policy_id

        # Register the environment before loading checkpoint
        self._register_nuel_env()

        # Load the trained algorithm and get the RLModule
        self.algorithm = Algorithm.from_checkpoint(checkpoint_path)
        self.rl_module = self.algorithm.get_module(self.policy_id)
        self.action_dist_cls = self.rl_module.get_inference_action_dist_cls()
        self._initialized = True

    def choose_target(self, me: "Player", players: List["Player"], observation=None) -> tuple[Optional["Player"], Optional[int]]:
        """Choose a target using the RLModule's forward_inference method"""

        # Create observation
        if observation is None and self.observation_model:
            observation = self.observation_model.create_observation(me, players)
        elif observation is None:
            # Fallback observation if no model provided
            observation = [me.accuracy] + [p.accuracy if p.alive else 0.0 for p in players if p != me]

        # Convert observation to tensor with batch dimension
        obs_tensor = torch.tensor([observation], dtype=torch.float32)

        # Use RLModule for inference
        fwd_inputs = {"obs": obs_tensor}
        fwd_outputs = self.rl_module.forward_inference(fwd_inputs)

        # Get action distribution and sample action
        action_dist = self.action_dist_cls.from_logits(fwd_outputs["action_dist_inputs"])
        action_tensor = action_dist.sample()

        # Convert to integer action (remove batch dimension)
        action = int(action_tensor[0].item())

        # Get valid targets
        if self.observation_model:
            targets = self.observation_model.get_targets(me, players)
        else:
            targets = [p for p in players if p != me and p.alive]

        # Return the chosen target
        if 0 <= action < len(targets):
            return targets[action], action
        return None, None

    def _register_nuel_env(self):
        """Register the Nuel environment with RLlib (same as in train.py)"""
        try:
            import ray
            from ray.tune.registry import register_env
            from rllib_marl.environment import NuelMultiAgentEnv

            # Initialize Ray if not already initialized (with warning suppression)
            if not ray.is_initialized():
                ray.init(
                    local_mode=True, 
                    ignore_reinit_error=True,
                    logging_level="ERROR",  # Suppress INFO warnings
                    log_to_driver=False     # Don't log to driver to reduce output
                )

            # Register the environment (suppress overriding warning)
            try:
                register_env("nuel_env", lambda config: NuelMultiAgentEnv(config))
            except Exception:
                # Environment already registered, ignore
                pass
        except Exception as e:
            print(f"Warning: Could not register nuel_env: {e}")
            raise

    def get_algorithm(self):
        """Get the underlying RLlib algorithm"""
        return self.algorithm