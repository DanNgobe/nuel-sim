# Import warning suppression first
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import suppress_warnings
suppress_warnings.suppress_ray_warnings()

import ray
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from rllib_marl.environment import NuelMultiAgentEnv
from rllib_marl.config import get_dqn_config, get_ppo_config, DEFAULT_CHECKPOINT_DIR
from pprint import pprint
import argparse

def register_nuel_env():
    """Register the Nuel environment with RLlib"""
    register_env("nuel_env", lambda config: NuelMultiAgentEnv(config))


def train_agent(algorithm="dqn", episodes=2000, checkpoint_dir=None):
    """Train RL agent with specified algorithm"""
    register_nuel_env()
    
    # Get configuration based on algorithm
    if algorithm == "dqn":
        config = get_dqn_config()
    elif algorithm == "ppo":
        config = get_ppo_config()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Choose 'dqn' or 'ppo'")
    
    # Build algorithm
    algo = config.build_algo()
    
    # Set up checkpoint directory
    checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    for i in range(episodes):
        result = algo.train()
        
        # Print progress
        if i < 100 or i % 100 == 0:
            print(f"Iteration {i}:")
            print(f"  Episode reward mean: {result.get('env_runners', {}).get('episode_reward_mean', 'N/A')}")
            print(f"  Episode length mean: {result.get('env_runners', {}).get('episode_len_mean', 'N/A')}")
            
            # Print per-policy rewards if available
            if "policy_reward_mean" in result:
                print(f"  Policy rewards: {result['policy_reward_mean']}")
        
        # Save checkpoint periodically
        if i % 500 == 0 and i > 0:
            checkpoint = algo.save_checkpoint(checkpoint_dir)
            print(f"Checkpoint saved at iteration {i}: {checkpoint}")
    
    # Save final checkpoint
    final_checkpoint = algo.save_checkpoint(checkpoint_dir)
    print(f"Training completed.")
    algo.stop()
    
    return final_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Train RLlib agent")
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()
    
    ray.init(local_mode=True)
    
    try:
        train_agent(
            algorithm=args.algorithm,
            episodes=args.episodes,
            checkpoint_dir=args.checkpoint_dir
        )
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()