# main.py
# Import warning suppression first
import suppress_warnings
suppress_warnings.suppress_ray_warnings()

import math
from core.game_manager import GameManager
from visual import run_game_visual, run_infinite_game_visual
import config.config_loader as config
from config.factories import create_game_objects

def print_detailed_episode_results(episode_data):
    """Print detailed results of an episode with round-by-round breakdown"""
    print("\n" + "="*60)
    print("EPISODE SUMMARY")
    print("="*60)
    
    if not episode_data:
        print("No episode data available.")
        return
    
    total_rounds = len(episode_data)
    print(f"Total Rounds: {total_rounds}")
    
    for round_idx, (observations, rewards, done, info) in enumerate(episode_data, 1):
        print(f"\n--- ROUND {round_idx} ---")
        
        # Print the shooting history for this round
        if "history" in info and info["history"]:
            print("Actions taken:")
            for shooter, target, hit in info["history"]:
                if shooter and target:
                    hit_status = "HIT" if hit else "MISS"
                    print(f"  P{shooter.id} → P{target.id}: {hit_status}")
                else:
                    print("  No valid action recorded")
        else:
            print("  No actions taken this round")
        
        # Print rewards for this round
        if rewards:
            print("Rewards earned:")
            for player_id, reward in rewards.items():
                if reward != 0:
                    print(f"  Player {player_id}: {reward:+.1f}")

        # Print current game state
        alive_count = info.get("alive_players", 0)
        print(f"Players still alive: {alive_count}")
        
        if done:
            print("🎯 GAME OVER!")
            break
    
    print("\n" + "="*60)
    print("EPISODE COMPLETE")
    print("="*60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Nuel Sim")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained model checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config.load_config(args.config)
        print(f"Loaded configuration from: {args.config}")

    # Create game objects
    game_objects = create_game_objects()

      # Override model path if provided
    if args.model_path:
        # Update strategies that use RLlib
        for i, strategy_type in enumerate(game_objects['strategies']):
            if hasattr(strategy_type, 'checkpoint_path'):
                strategy_type.checkpoint_path = args.model_path
                print(f"Using model from: {args.model_path}")
                
    cfg = config.get_config()
    
    game_manager = GameManager(
        num_players=cfg['game']['num_players'],
        gameplay=game_objects['gameplay'],
        observation_model=game_objects['observation_model'],
        max_rounds=cfg['game']['num_rounds'],
        marksmanship_range=tuple(cfg['players']['marksmanship_range']),
        strategies=game_objects['strategies'],
        assigned_accuracies=cfg['players']['accuracies'],
        has_ghost=cfg['gameplay']['has_ghost'],
        screen_width=cfg['visual']['screen_width'],
        screen_height=cfg['visual']['screen_height']
    )
    
    if cfg['game']['run_mode'] == "visualize":
        print("Running visual simulation...")
        run_infinite_game_visual(game_manager)
    else:
        print("Running single episode...")
        result = game_manager.run_episode()
        print_detailed_episode_results(result)

if __name__ == "__main__":
    main()