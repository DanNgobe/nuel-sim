# Import warning suppression first
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import suppress_warnings
suppress_warnings.suppress_ray_warnings()

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env
from rllib_marl.environment import NuelMultiAgentEnv
from pprint import pprint
import argparse
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

# Import your custom modules
import config.settings as config
from config.factories import create_game_objects

DEFAULT_CHECKPOINT_DIR = config.RLLIB_CHECKPOINT_PATH

def get_env_config():
    """Get environment configuration for RLlib"""
    game_objects = create_game_objects()
    return {
        "num_players": config.NUM_PLAYERS,
        "gameplay": game_objects['gameplay'],
        "observation_model": game_objects['observation_model'],
        "max_rounds": config.NUM_ROUNDS,
        "marksmanship_range": config.MARKSMANSHIP_RANGE,
        "strategies": [],  # Empty for RL training
        "assigned_accuracies": getattr(config, 'ASSIGNED_ACCURACIES', []),
        "has_ghost": config.HAS_GHOST,
        "capture_all_terminate": False
    }

def get_ppo_config():
    """Get PPO configuration for shared policy"""
    env_config = get_env_config()
    
    return (
        PPOConfig()
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=True,
                   enable_env_runner_and_connector_v2=True)
        .environment("nuel_env", env_config=env_config)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
            policies_to_train=["shared_policy"],
        )
        .env_runners(
            # ↓↓↓ Use 0 remote workers to avoid overhead (runs on local CPU/GPU)
            num_env_runners=0,
            # ↓↓↓ Minimize environments to reduce total samples collected per step
            num_envs_per_env_runner=1,
            # Add episode handling configuration
            rollout_fragment_length="auto",
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(rl_module_specs={
                "shared_policy": RLModuleSpec(),
            }),
            model_config=DefaultModelConfig(
                fcnet_hiddens=[128, 128],
                use_lstm=False,  # Disabled LSTM to fix indexing issue
            ),
        )
        .training(
            gamma=0.98,
            train_batch_size=256
        )
        # .resources(num_gpus=0, num_cpus_for_main_process=1)
    )

def get_dqn_config():
    env_config = get_env_config()

    return (
        DQNConfig()
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=True,
                   enable_env_runner_and_connector_v2=True)
        .environment("nuel_env", env_config=env_config)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
            policies_to_train=["shared_policy"],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(rl_module_specs={
                "shared_policy": RLModuleSpec(),
            })
        )
        .training(
            gamma=0.98,
            lr=0.0005,
            train_batch_size=256,
            target_network_update_freq=500,
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 50000,
            }
        )
    )
