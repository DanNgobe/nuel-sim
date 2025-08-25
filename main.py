# main.py
import math
from core.game_manager import GameManager
from visual import run_game_visual
import config

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
    # Create game manager
    game_manager = GameManager(
        num_players=config.NUM_PLAYERS,
        gameplay=config.GAME_PLAY,
        observation_model=config.OBSERVATION_MODEL,
        max_rounds=config.NUM_ROUNDS,
        marksmanship_range=config.MARKSMANSHIP_RANGE,
        strategies=config.ASSIGNED_STRATEGIES,
        has_ghost=config.HAS_GHOST,
        screen_width=config.SCREEN_WIDTH,
        screen_height=config.SCREEN_HEIGHT
    )
    
    if config.RUN_MODE == "visualize":
        print("Running visual simulation...")
        game = game_manager.reset_game()
        run_game_visual(game)
    
    else:  # single episode
        print("Running single episode...")
        result = game_manager.run_episode()
        print_detailed_episode_results(result)

if __name__ == "__main__":
    main()