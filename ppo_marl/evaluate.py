# ppo_marl/evaluate.py
from core.game_manager import GameManager
from ppo_marl.strategy import PPOStrategy
from ppo_marl.settings import get_model_path
from config.factories import create_game_objects
import config
import argparse
import os

def main(model_path=None, episodes=100):
    # Create game objects using factory
    game_objects = create_game_objects()
    
    # Get model path if not provided
    if model_path is None:
        model_path = get_model_path(
            config.NUM_PLAYERS, 
            config.GAME_PLAY_TYPE,
            game_objects['observation_model'].name
        )
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a model first using: python -m ppo_marl.train")
        return
    
    # Create PPO strategy for evaluation (no exploration)
    ppo_strategy = PPOStrategy(
        game_objects['observation_model'],
        model_path=model_path, 
        explore=False  # No exploration during evaluation
    )
    
    # All players use the same PPO strategy
    ppo_strategies = [ppo_strategy for _ in range(config.NUM_PLAYERS)]
    
    # Create GameManager with PPO strategies
    game_manager = GameManager(
        num_players=config.NUM_PLAYERS,
        gameplay=game_objects['gameplay'],
        observation_model=game_objects['observation_model'],
        max_rounds=config.NUM_ROUNDS,
        marksmanship_range=config.MARKSMANSHIP_RANGE,
        strategies=ppo_strategies,
        assigned_accuracies=config.ASSIGNED_ACCURACIES,
        has_ghost=config.HAS_GHOST,
        screen_width=config.SCREEN_WIDTH,
        screen_height=config.SCREEN_HEIGHT
    )
    
    print(f"Evaluating PPO model: {model_path}")
    print(f"Running {episodes} episodes...")
    
    # Statistics
    wins_by_position = [0] * config.NUM_PLAYERS
    wins_by_accuracy = [0] * config.NUM_PLAYERS
    total_rounds = 0
    
    for episode in range(episodes):
        observations, infos = game_manager.reset()
        rounds_played = 0
        
        while True:
            actions = {}
            for agent_id, obs in observations.items():
                action, _, _ = ppo_strategy.get_agent().act(obs, explore=False)
                actions[agent_id] = action
            
            next_observations, rewards, terminateds, truncateds, next_infos = game_manager.step(actions)
            rounds_played += 1
            observations = next_observations
            
            if terminateds.get("__all__", False):
                break
        
        total_rounds += rounds_played
        
        # Record winner statistics
        alive_players = [p for p in game_manager.current_game.players if p.alive]
        if len(alive_players) == 1:
            winner = alive_players[0]
            wins_by_position[winner.id] += 1
            
            # Win by accuracy ranking
            sorted_players = sorted(game_manager.current_game.players, key=lambda p: p.accuracy)
            accuracy_rankings = {p.id: rank for rank, p in enumerate(sorted_players)}
            winner_rank = accuracy_rankings[winner.id]
            wins_by_accuracy[winner_rank] += 1
        
        if (episode + 1) % 20 == 0:
            print(f"Completed {episode + 1}/{episodes} episodes")
    
    # Print results
    print(f"\n=== Evaluation Results ({episodes} episodes) ===")
    print(f"Average game length: {total_rounds / episodes:.1f} rounds")
    
    print(f"\nWins by starting position:")
    for pos in range(config.NUM_PLAYERS):
        win_rate = wins_by_position[pos] / episodes * 100
        print(f"  Position {pos}: {wins_by_position[pos]} wins ({win_rate:.1f}%)")
    
    print(f"\nWins by accuracy ranking:")
    accuracy_labels = ['Weakest', '2nd Strongest', '3rd Strongest', '4th Strongest', 'Strongest']
    for rank in range(config.NUM_PLAYERS):
        win_rate = wins_by_accuracy[rank] / episodes * 100
        label = accuracy_labels[rank] if rank < len(accuracy_labels) else f'Rank {rank}'
        print(f"  {label}: {wins_by_accuracy[rank]} wins ({win_rate:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent.")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate (default: 100)")
    args = parser.parse_args()

    main(model_path=args.model_path, episodes=args.episodes)