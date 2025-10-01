# Import warning suppression first
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import suppress_warnings
suppress_warnings.suppress_ray_warnings()

import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
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

def get_dqn_config():
    """Get DQN configuration for shared policy"""
    env_config = get_env_config()
    
    return (
        DQNConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .environment("nuel_env", env_config=env_config)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
            policies_to_train=["shared_policy"],
        ).env_runners(
            num_envs_per_env_runner=1,
            # Add episode handling configuration
            rollout_fragment_length="auto",
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(rl_module_specs={
                "shared_policy": RLModuleSpec(),
            })
        )
        .training(
            train_batch_size_per_learner=4000,
            lr=1e-4,
            gamma=0.99,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            # DQN-specific settings
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 100000,
                "alpha": 0.6,
                "beta": 0.4,
            },
            num_steps_sampled_before_learning_starts=1000,
            target_network_update_freq=1000,
            n_step=3,
            double_q=True,
            dueling=True,
        )
    )

def get_ppo_config():
    """Get PPO configuration for shared policy"""
    env_config = get_env_config()
    
    return (
        PPOConfig()
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
                use_lstm=True,
                max_seq_len=20,
                lstm_use_prev_action=False,
                lstm_use_prev_reward=False,
            ),
        )
        .training(
            gamma=0.99,
            lr=5e-4,
            train_batch_size=4000
        )
        # .resources(num_gpus=0, num_cpus_for_main_process=1)
    )
