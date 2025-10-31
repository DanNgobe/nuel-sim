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
import config.config_loader as config
from config.factories import create_game_objects

def get_default_checkpoint_dir():
    cfg = config.get_config()
    algorithm = cfg['rllib']['algorithm']
    base_path = cfg['rllib']['checkpoint_base_path']
    num_players = cfg['game']['num_players']
    gameplay_type = cfg['gameplay']['type']
    obs_model_type = cfg['observation']['model_type']
    return os.path.abspath(f"{base_path}/{algorithm}/{num_players}_players/{gameplay_type}_{obs_model_type}")

DEFAULT_CHECKPOINT_DIR = get_default_checkpoint_dir()

def get_env_config():
    """Get environment configuration for RLlib"""
    game_objects = create_game_objects()
    cfg = config.get_config()
    return {
        "num_players": cfg['game']['num_players'],
        "gameplay": game_objects['gameplay'],
        "observation_model": game_objects['observation_model'],
        "max_rounds": cfg['game']['num_rounds'],
        "marksmanship_range": tuple(cfg['players']['marksmanship_range']),
        "strategies": [],  # Empty for RL training
        "assigned_accuracies": cfg['players']['accuracies'],
        "has_ghost": cfg['gameplay']['has_ghost'],
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
