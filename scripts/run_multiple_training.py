#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_multiple_training(num_runs=5, episodes=4000, base_output_dir="multiple_runs", config_path=None):
    """Run multiple training sessions with different seeds"""
    
    # Create base output directory
    base_path = Path(base_output_dir)
    base_path.mkdir(exist_ok=True)
    
    print(f"Running {num_runs} training sessions...")
    
    for run_id in range(num_runs):
        print(f"\n=== Starting Run {run_id + 1}/{num_runs} ===")
        
        # Create run-specific output directory
        run_dir = base_path / f"run_{run_id:02d}"
        run_dir.mkdir(exist_ok=True)
        
        # Build command with config and default flags
        cmd = [
            sys.executable, "-m", "rllib_marl.train",
            "--algorithm", "ppo",
            "--episodes", str(episodes),
            "--output-path", str(run_dir),
            "--plot",  # Always include plot
            "--evaluate"  # Always include evaluate
        ]
        
        # Add config if provided
        if config_path:
            cmd.extend(["--config", config_path])
        
        # Set environment variable for seed (Ray uses this)
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(run_id * 42)
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            print(f"Run {run_id + 1} completed successfully")
            
            # Save run log
            with open(run_dir / "training_log.txt", "w") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Seed: {run_id * 42}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"STDOUT:\n{result.stdout}\n")
                f.write(f"STDERR:\n{result.stderr}\n")
                
        except subprocess.CalledProcessError as e:
            print(f"Run {run_id + 1} failed with error: {e}")
            
            # Save error log
            with open(run_dir / "error_log.txt", "w") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Seed: {run_id * 42}\n")
                f.write(f"Return code: {e.returncode}\n")
                f.write(f"STDOUT:\n{e.stdout}\n")
                f.write(f"STDERR:\n{e.stderr}\n")
    
    print(f"\nAll training runs completed. Results saved in: {base_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple training sessions")
    parser.add_argument("--runs", type=int, default=5, help="Number of training runs")
    parser.add_argument("--episodes", type=int, default=4000, help="Episodes per run")
    parser.add_argument("--output-dir", type=str, default="multiple_runs", help="Base output directory")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    
    args = parser.parse_args()
    
    run_multiple_training(
        num_runs=args.runs,
        episodes=args.episodes,
        base_output_dir=args.output_dir,
        config_path=args.config
    )