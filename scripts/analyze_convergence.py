#!/usr/bin/env python3

import os
import sys
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, mannwhitneyu
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config.config_loader as config
from config.factories import create_game_objects
from core.game_manager import GameManager
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.registry import register_env
from rllib_marl.environment import NuelMultiAgentEnv
from rllib_marl.config import get_ppo_config
import torch

def collect_action_probabilities(checkpoint_path, num_episodes=100, config_path=None):
    """Collect action probabilities across different game states"""
    
    # Load config if provided
    if config_path:
        config.load_config(config_path)
        print(f"Loaded configuration from: {config_path}")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Register the environment before loading checkpoint
    register_env("nuel_env", lambda env_config: NuelMultiAgentEnv(env_config))
    
    # Load trained model - ensure absolute path
    checkpoint_path = Path(checkpoint_path).resolve()
    
    # Build a fresh algorithm to get the right structure
    ppo_config = get_ppo_config()
    algo = ppo_config.build()
    
    # Try to restore just the learner/module state (the actual trained weights)
    try:
        learner_dir = checkpoint_path / "learner_group" / "learner"
        if learner_dir.exists():
            # Load the module state directly - check for the rl_module subdirectory
            rl_module_dir = learner_dir / "rl_module" / "shared_policy"
            if rl_module_dir.exists() and (rl_module_dir / "module_state.pkl").exists():
                print(f"Loading trained weights from {rl_module_dir}")
                # Load the module state pickle file
                import pickle
                with open(rl_module_dir / "module_state.pkl", "rb") as f:
                    module_state = pickle.load(f)
                
                # Get the RLModule and set its state
                rl_module = algo.get_module("shared_policy")
                rl_module.set_state(module_state)
                print("Successfully loaded trained weights")
            else:
                print("Warning: No module state found, using random initialization")
        else:
            print("Warning: No learner directory found, using random initialization")
    except Exception as e:
        print(f"Warning: Failed to restore weights: {e}")
        import traceback
        traceback.print_exc()
        print("Using freshly initialized model")
    
    # Create game environment - use_random_for_training=True avoids creating RLlibStrategy
    game_objects = create_game_objects(use_random_for_training=True)
    cfg = config.get_config()
    
    game_manager = GameManager(
        num_players=cfg['game']['num_players'],
        gameplay=game_objects['gameplay'],
        observation_model=game_objects['observation_model'],
        max_rounds=cfg['game']['num_rounds'],
        marksmanship_range=tuple(cfg['players']['marksmanship_range']),
        strategies=[],  # Empty - we're using the algo directly
        assigned_accuracies=cfg['players']['accuracies'],
        has_ghost=cfg['gameplay']['has_ghost']
    )
    
    action_probs = []
    
    for episode in range(num_episodes):
        obs, _ = game_manager.reset()
        
        while obs:  # While there are active agents
            episode_actions = {}
            
            for agent_id, observation in obs.items():
                # Get action probabilities from RLModule (new API)
                # Convert observation to batch format
                obs_batch = {
                    "obs": torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                }
                
                # Get the RLModule for the shared policy
                rl_module = algo.get_module("shared_policy")
                
                # Forward pass through the module
                with torch.no_grad():
                    output = rl_module.forward_inference(obs_batch)
                    
                    # Extract action distribution logits
                    if "action_dist_inputs" in output:
                        logits = output["action_dist_inputs"].squeeze(0).cpu().numpy()
                        # Convert logits to probabilities using softmax
                        probs = np.exp(logits - np.max(logits))  # Numerical stability
                        probs = probs / np.sum(probs)
                        action_probs.append(probs)
            
            # Take random actions to continue episode
            actions = {agent_id: np.random.randint(0, game_manager.action_spaces[agent_id].n) 
                      for agent_id in obs.keys()}
            
            obs, _, terminated, _, _ = game_manager.step(actions)
            
            if terminated.get("__all__", False):
                break
    
    ray.shutdown()
    return np.array(action_probs)

def compute_kl_divergence(probs1, probs2):
    """Compute KL divergence between two probability distributions"""
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    probs1 = probs1 + eps
    probs2 = probs2 + eps
    
    # Normalize
    probs1 = probs1 / np.sum(probs1)
    probs2 = probs2 / np.sum(probs2)
    
    return entropy(probs1, probs2)

def analyze_convergence(runs_dir="multiple_runs", config_path=None):
    """Analyze convergence across multiple training runs"""
    
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        print(f"Directory {runs_dir} not found")
        return
    
    # Try to auto-detect config from error log if not provided
    if not config_path:
        first_run = runs_path / "run_00"
        error_log = first_run / "error_log.txt"
        if error_log.exists():
            with open(error_log, 'r') as f:
                for line in f:
                    if '--config' in line:
                        # Extract config path from command line
                        parts = line.split('--config')
                        if len(parts) > 1:
                            config_path = parts[1].strip().split()[0]
                            print(f"Auto-detected config: {config_path}")
                            break
    
    # Find all run directories with checkpoints
    run_dirs = []
    for run_dir in runs_path.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            # Check if this directory contains checkpoint files directly
            if (run_dir / "rllib_checkpoint.json").exists():
                run_dirs.append((run_dir.name, run_dir))
            else:
                # Check for checkpoints subdirectory (legacy structure)
                checkpoint_dir = run_dir / "checkpoints"
                if checkpoint_dir.exists():
                    checkpoints = list(checkpoint_dir.glob("checkpoint_*"))
                    if checkpoints:
                        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("_")[1]))
                        run_dirs.append((run_dir.name, latest_checkpoint))
    
    if len(run_dirs) < 2:
        print(f"Need at least 2 runs with checkpoints, found {len(run_dirs)}")
        return
    
    print(f"Analyzing {len(run_dirs)} training runs...")
    
    # Collect action probabilities for each run
    all_probs = {}
    for run_name, checkpoint_path in run_dirs:
        print(f"Collecting probabilities for {run_name}...")
        probs = collect_action_probabilities(str(checkpoint_path), config_path=config_path)
        all_probs[run_name] = probs
        print(f"Collected {len(probs)} probability samples")
    
    # Compute pairwise KL divergences
    run_names = list(all_probs.keys())
    kl_matrix = np.zeros((len(run_names), len(run_names)))
    
    for i, run1 in enumerate(run_names):
        for j, run2 in enumerate(run_names):
            if i != j:
                # Average probabilities across all samples
                avg_probs1 = np.mean(all_probs[run1], axis=0)
                avg_probs2 = np.mean(all_probs[run2], axis=0)
                
                kl_div = compute_kl_divergence(avg_probs1, avg_probs2)
                kl_matrix[i, j] = kl_div
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap of KL divergences
    im = ax1.imshow(kl_matrix, cmap='viridis')
    ax1.set_xticks(range(len(run_names)))
    ax1.set_yticks(range(len(run_names)))
    ax1.set_xticklabels(run_names, rotation=45)
    ax1.set_yticklabels(run_names)
    ax1.set_title("KL Divergence Between Runs")
    plt.colorbar(im, ax=ax1)
    
    # Add text annotations
    for i in range(len(run_names)):
        for j in range(len(run_names)):
            text = ax1.text(j, i, f'{kl_matrix[i, j]:.3f}',
                           ha="center", va="center", color="white")
    
    # Distribution of KL divergences
    upper_triangle = kl_matrix[np.triu_indices_from(kl_matrix, k=1)]
    ax2.hist(upper_triangle, bins=10, alpha=0.7, color='skyblue')
    ax2.set_xlabel("KL Divergence")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of KL Divergences")
    ax2.axvline(np.mean(upper_triangle), color='red', linestyle='--', 
                label=f'Mean: {np.mean(upper_triangle):.3f}')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save results
    output_path = runs_path / "convergence_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Analysis saved to: {output_path}")
    
    # Save numerical results
    results = {
        'kl_matrix': kl_matrix,
        'run_names': run_names,
        'mean_kl_divergence': np.mean(upper_triangle),
        'std_kl_divergence': np.std(upper_triangle)
    }
    
    with open(runs_path / "convergence_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Test if KL divergences are significantly different from random
    # Generate random probability distributions for comparison
    n_random = 1000
    action_dim = all_probs[run_names[0]].shape[1]
    random_kls = []
    
    for _ in range(n_random):
        # Generate two random probability distributions
        rand1 = np.random.dirichlet(np.ones(action_dim))
        rand2 = np.random.dirichlet(np.ones(action_dim))
        random_kl = compute_kl_divergence(rand1, rand2)
        random_kls.append(random_kl)
    
    # Mann-Whitney U test
    statistic, p_value = mannwhitneyu(upper_triangle, random_kls, alternative='less')
    
    # Print summary
    print(f"\n=== Convergence Analysis Summary ===")
    print(f"Mean KL Divergence: {np.mean(upper_triangle):.4f}")
    print(f"Std KL Divergence: {np.std(upper_triangle):.4f}")
    print(f"Min KL Divergence: {np.min(upper_triangle):.4f}")
    print(f"Max KL Divergence: {np.max(upper_triangle):.4f}")
    print(f"Mean Random KL Divergence: {np.mean(random_kls):.4f}")
    print(f"Statistical test (vs random): p = {p_value:.4f}")
    
    if p_value < 0.05:
        print("✓ KL divergences are significantly LOWER than random (convergence detected)")
    else:
        print("~ KL divergences not significantly different from random")
    
    if np.mean(upper_triangle) < 0.1:
        print("✓ Models show HIGH convergence (similar strategies)")
    elif np.mean(upper_triangle) < 0.5:
        print("~ Models show MODERATE convergence")
    else:
        print("✗ Models show LOW convergence (diverse strategies)")
    
    # Add statistical test result to plot
    sig_text = f"vs Random: p = {p_value:.3f}" + (" *" if p_value < 0.05 else "")
    ax2.text(0.02, 0.98, sig_text, transform=ax2.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze strategy convergence")
    parser.add_argument("--runs-dir", type=str, default="multiple_runs", 
                       help="Directory containing multiple training runs")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Episodes to sample for probability collection")
    parser.add_argument("--config", type=str, default=None,
                       help="Configuration file used for training (will auto-detect if not provided)")
    
    args = parser.parse_args()
    analyze_convergence(args.runs_dir, config_path=args.config)