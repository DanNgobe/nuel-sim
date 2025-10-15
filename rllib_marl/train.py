# Import warning suppression first
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import suppress_warnings
suppress_warnings.suppress_ray_warnings()

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
from ray.tune.registry import register_env
from rllib_marl.environment import NuelMultiAgentEnv
from rllib_marl.config import get_ppo_config, get_dqn_config, DEFAULT_CHECKPOINT_DIR
from pprint import pprint
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def register_nuel_env():
    """Register the Nuel environment with RLlib"""
    register_env("nuel_env", lambda config: NuelMultiAgentEnv(config))


def plot_training_stats(stats, save_path="rllib_marl/training_stats.png"):
    """Plot training statistics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode reward mean
    if 'episode_reward_mean' in stats and len(stats['episode_reward_mean']) > 0:
        ax1.plot(stats['episode_reward_mean'], linewidth=2, color='green')
        ax1.set_title('Episode Reward Mean')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Mean Reward')
        ax1.grid(True, alpha=0.3)
    
    # Episode length mean
    if 'episode_len_mean' in stats and len(stats['episode_len_mean']) > 0:
        ax2.plot(stats['episode_len_mean'], linewidth=2, color='blue')
        ax2.set_title('Episode Length Mean')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Mean Length')
        ax2.grid(True, alpha=0.3)
    
    # Policy loss (if available)
    if 'policy_loss' in stats and len(stats['policy_loss']) > 0:
        ax3.plot(stats['policy_loss'], linewidth=2, color='red')
        ax3.set_title('Policy Loss')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Loss')
        ax3.grid(True, alpha=0.3)
    
    # Value function loss (if available)
    if 'vf_loss' in stats and len(stats['vf_loss']) > 0:
        ax4.plot(stats['vf_loss'], linewidth=2, color='orange')
        ax4.set_title('Value Function Loss')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Loss')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining plots saved to: {save_path}")
    plt.close()


def train_agent(algorithm="dqn", episodes=2000, checkpoint_dir=None, plot_stats=False, evaluate_after=False):
    """Train RL agent with specified algorithm"""
    register_nuel_env()
    
    # Get configuration based on algorithm
    if algorithm == "ppo":
        config = get_ppo_config()
    elif algorithm == "dqn":
        config = get_dqn_config()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Choose 'ppo' or 'dqn'")
    
    # Build algorithm
    algo = config.build_algo()
    
    # Set up checkpoint directory
    checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Statistics collection
    stats = defaultdict(list) if plot_stats else None
    
    # Training loop
    for i in range(episodes):
        result = algo.train()
        
        # Collect statistics
        if plot_stats:
            env_runners_data = result.get('env_runners', {})
            stats['episode_reward_mean'].append(env_runners_data.get('episode_return_mean', 0))
            stats['episode_len_mean'].append(env_runners_data.get('episode_len_mean', 0))
            
            # Get learner data for loss metrics
            learners_data = result.get('learners', {}).get('shared_policy', {})
            if 'policy_loss' in learners_data:
                stats['policy_loss'].append(float(learners_data['policy_loss']))
            if 'vf_loss' in learners_data:
                stats['vf_loss'].append(float(learners_data['vf_loss']))
        
        # Print progress
        if i < 100 or i % 100 == 0:
            env_runners_data = result.get('env_runners', {})
            print(f"Iteration {i}:")
            print(f"  Episode reward mean: {env_runners_data.get('episode_return_mean', 'N/A')}")
            print(f"  Episode length mean: {env_runners_data.get('episode_len_mean', 'N/A')}")
        
        # Save checkpoint periodically
        if i % 500 == 0 and i > 0:
            algo.save_checkpoint(checkpoint_dir)
            print(f"Checkpoint saved at iteration {i}")
    
    # Save final checkpoint
    algo.save_checkpoint(checkpoint_dir)
    print(f"\nTraining completed.")
    
    # Generate plots if requested
    if plot_stats and stats:
        stats_plot_path = os.path.join(checkpoint_dir, "training_stats.png")
        plot_training_stats(dict(stats), save_path=stats_plot_path)
    
    # Run evaluation script if requested
    if evaluate_after:
        from scripts.evaluate import evaluate
        import config.settings as config
        from config.factories import create_game_objects
        
        game_objects = create_game_objects()
        evaluate_output_dir = os.path.join(checkpoint_dir, f"evaluation/{game_objects['observation_model'].name}")
        os.makedirs(evaluate_output_dir, exist_ok=True)
        
        evaluate(num_episodes=1000, single_strategy='RLlibStrategy', output_path=evaluate_output_dir)
    
    algo.stop()


def main():
    parser = argparse.ArgumentParser(description="Train RLlib agent")
    parser.add_argument("--algorithm", choices=["sac", "ppo", "dqn"], default="dqn",
                        help="Algorithm to use for training (default: dqn)")
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Number of training iterations (default: 2000)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory to save checkpoints (default: from config)")
    parser.add_argument("--plot", action="store_true",
                        help="Collect statistics and generate plots at the end")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation after training")
    args = parser.parse_args()
    
    ray.init(local_mode=True)
    
    try:
        train_agent(
            algorithm=args.algorithm,
            episodes=args.episodes,
            checkpoint_dir=args.checkpoint_dir,
            plot_stats=args.plot,
            evaluate_after=args.evaluate
        )
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()