# dqn_marl/train.py
from core.game_manager import GameManager
from dqn_marl.replay_buffer import ReplayBuffer
from dqn_marl.strategy import DQNStrategy
from dqn_marl.settings import MEMORY_SIZE, get_model_path, BATCH_SIZE, TARGET_UPDATE_FREQUENCY
from config.factories import create_game_objects
import config
import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def moving_average(data, window_size):
    """Calculate moving average with given window size"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_training_stats(stats, episodes, save_path="dqn_marl/models/training_stats.png"):
    """Plot training statistics"""
    # Determine window size based on number of episodes
    window_size = max(10, episodes // 100)  # At least 10, or 1% of episodes
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Episode lengths
    episode_lengths = stats['episode_lengths']
    if len(episode_lengths) > 0:
        ax1.plot(episode_lengths, alpha=0.3, color='blue', label='Raw')
        if len(episode_lengths) >= window_size:
            smoothed = moving_average(episode_lengths, window_size)
            ax1.plot(range(window_size-1, len(episode_lengths)), smoothed, 
                    color='blue', linewidth=2, label=f'Moving Avg ({window_size})')
        ax1.set_title('Episode Lengths')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Game Length (Rounds)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Average rewards
    avg_rewards = stats['average_rewards']
    if len(avg_rewards) > 0:
        ax2.plot(avg_rewards, alpha=0.3, color='green', label='Raw')
        if len(avg_rewards) >= window_size:
            smoothed = moving_average(avg_rewards, window_size)
            ax2.plot(range(window_size-1, len(avg_rewards)), smoothed, 
                    color='green', linewidth=2, label=f'Moving Avg ({window_size})')
        ax2.set_title('Average Rewards per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Win rate by position
    position_wins = stats['position_wins']
    total_games = len(episode_lengths)
    if total_games > 0:
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for pos in range(config.NUM_PLAYERS):
            wins = position_wins[pos]
            win_rates = [sum(wins[:i+1])/(i+1) for i in range(len(wins))]
            if len(win_rates) >= window_size:
                smoothed = moving_average(win_rates, window_size)
                ax3.plot(range(window_size-1, len(win_rates)), smoothed, 
                        color=colors[pos % len(colors)], linewidth=2, 
                        label=f'Position {pos}')
            else:
                ax3.plot(win_rates, color=colors[pos % len(colors)], 
                        linewidth=2, label=f'Position {pos}')
        ax3.set_title('Win Rate by Starting Position')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Win Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
    
    # 4. Win rate by accuracy ranking
    accuracy_wins = stats['accuracy_wins']
    if total_games > 0:
        accuracy_labels = ['Weakest', '2nd Strongest', '3rd Strongest', '4th Strongest', 'Strongest']
        colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']

        for rank in range(config.NUM_PLAYERS):
            wins = accuracy_wins[rank]
            if len(wins) > 0:
                win_rates = [sum(wins[:i+1])/(i+1) for i in range(len(wins))]
                if len(win_rates) >= window_size:
                    smoothed = moving_average(win_rates, window_size)
                    ax4.plot(range(window_size-1, len(win_rates)), smoothed, 
                            color=colors[rank % len(colors)], linewidth=2, 
                            label=accuracy_labels[rank] if rank < len(accuracy_labels) else f'Rank {rank}')
                else:
                    ax4.plot(win_rates, color=colors[rank % len(colors)], 
                            linewidth=2, 
                            label=accuracy_labels[rank] if rank < len(accuracy_labels) else f'Rank {rank}')
        ax4.set_title('Win Rate by Accuracy Ranking')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Win Rate')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training plots saved to: {save_path}")
    plt.show()

def main(episodes=2000, plot_stats=False, evaluate_after=False):
    # Create game objects using factory
    game_objects = create_game_objects()
    
    # Get model path
    model_path = get_model_path(
        config.NUM_PLAYERS, 
        config.GAME_PLAY_TYPE,
        game_objects['observation_model'].name
    )
    
    # Create single shared DQN strategy and agent for all players
    dqn_strategy = DQNStrategy(
        game_objects['observation_model'],
        model_path=model_path, 
        explore=True
    )
    
    # All players use the same DQN strategy (shared agent)
    dqn_strategies = [dqn_strategy for _ in range(config.NUM_PLAYERS)]
    
    # Single shared agent and replay buffer
    agent = dqn_strategy.get_agent()
    replay_buffer = ReplayBuffer(MEMORY_SIZE)
    
    # Create GameManager with DQN strategies
    game_manager = GameManager(
        num_players=config.NUM_PLAYERS,
        gameplay=game_objects['gameplay'],
        observation_model=game_objects['observation_model'],
        max_rounds=config.NUM_ROUNDS,
        marksmanship_range=config.MARKSMANSHIP_RANGE,
        strategies=dqn_strategies,
        assigned_accuracies=config.ASSIGNED_ACCURACIES,
        has_ghost=config.HAS_GHOST,
        screen_width=config.SCREEN_WIDTH,
        screen_height=config.SCREEN_HEIGHT
    )
    
    # Initialize statistics collection
    stats = {
        'episode_lengths': [],
        'average_rewards': [],
        'position_wins': [[] for _ in range(config.NUM_PLAYERS)],  # wins for each starting position
        'accuracy_wins': [[] for _ in range(config.NUM_PLAYERS)]   # wins for each accuracy rank
    }
    
    # Training loop
    print(f"Starting DQN MARL training for {episodes} episodes...")
    print(f"Model will be saved to: {model_path}")
    if plot_stats:
        print("Statistics will be collected and plotted at the end.")
    
    # Step counter for target network updates
    total_steps = 0
    
    for episode in range(episodes):
        observations, infos = game_manager.reset()
        
        prev_observations = {}
        prev_actions = {}
        
        # Episode statistics
        episode_rewards = []
        rounds_played = 0
        
        while True:
            actions = {}
            for agent_id, obs in observations.items():
                action = agent.act(obs, explore=True)
                actions[agent_id] = action
                
                # Store for experience replay
                prev_observations[agent_id] = obs.copy()
                prev_actions[agent_id] = action
            
            # Execute actions and get results
            next_observations, rewards, terminateds, truncateds, next_infos = game_manager.step(actions)
            
            # Collect episode rewards for statistics
            if plot_stats:
                for agent_id, reward in rewards.items():
                    episode_rewards.append(reward)
            
            rounds_played += 1
            
            # Store experiences and update shared agent
            for agent_id in actions.keys():
                if agent_id in rewards:
                    reward = rewards[agent_id]
                    
                    # Determine if this agent's episode is done
                    done = terminateds.get("__all__", False) or terminateds.get(agent_id, False)
                    
                    # Get next observation for this agent (if available)
                    next_obs = next_observations.get(agent_id, game_manager.get_player_observation(agent_id))
                    
                    # Store experience in shared replay buffer
                    replay_buffer.push(
                        prev_observations[agent_id],
                        prev_actions[agent_id], 
                        reward,
                        next_obs,
                        done
                    )
            
            # Update shared agent if enough experiences collected
            if len(replay_buffer) >= BATCH_SIZE:
                agent.update(replay_buffer)
                total_steps += 1
                
                # Update target network periodically for DDQN stability
                if total_steps % TARGET_UPDATE_FREQUENCY == 0:
                    agent.update_target_network()
                    print(f"Target network updated at step {total_steps}")
            
            # Update observations for next iteration
            observations = next_observations
            
            # Check if game is over
            if terminateds.get("__all__", False):
                break
        
        # Collect episode statistics
        if plot_stats:
            # Episode length
            stats['episode_lengths'].append(rounds_played)
            
            # Average reward for this episode
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            stats['average_rewards'].append(avg_reward)
            
            # Winner and position stats
            alive_players = [p for p in game_manager.current_game.players if p.alive]
            if len(alive_players) == 1:
                winner = alive_players[0]
                
                # Win by starting position
                for pos in range(config.NUM_PLAYERS):
                    is_winner = (winner.id == pos)
                    stats['position_wins'][pos].append(1 if is_winner else 0)
                
                # Win by accuracy ranking
                # Sort players by accuracy to get rankings
                sorted_players = sorted(game_manager.current_game.players, key=lambda p: p.accuracy)
                sorted_players = [p for p in sorted_players if p.name != "Ghost"]
                accuracy_rankings = {p.id: rank for rank, p in enumerate(sorted_players)}
                
                winner_rank = accuracy_rankings[winner.id]
                for rank in range(config.NUM_PLAYERS):
                    is_winner = (winner_rank == rank)
                    stats['accuracy_wins'][rank].append(1 if is_winner else 0)
            else:
                # No clear winner (tie or error)
                for pos in range(config.NUM_PLAYERS):
                    stats['position_wins'][pos].append(0)
                for rank in range(config.NUM_PLAYERS):
                    stats['accuracy_wins'][rank].append(0)
        
        # Update epsilon for shared agent
        agent.decay_epsilon(episode, episodes)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} , Epsilon: {agent.epsilon:.3f}")
            if plot_stats and len(stats['episode_lengths']) > 0:
                recent_avg_length = np.mean(stats['episode_lengths'][-100:])
                recent_avg_reward = np.mean(stats['average_rewards'][-100:])
                print(f"  Recent Avg Game Length: {recent_avg_length:.1f}")
                print(f"  Recent Avg Reward: {recent_avg_reward:.2f}")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the shared model
    agent.save_model(model_path)
    print(f"Training completed and shared model saved to {model_path}")
    
    # Generate plots if requested
    if plot_stats:
        model_path_without_ext = os.path.splitext(model_path)[0]
        stats_plot_path = f"{model_path_without_ext}_training_stats.png"
        plot_training_stats(stats, episodes, save_path=stats_plot_path)
    
    # Run evaluation if requested
    if evaluate_after:
        from scripts.evaluate import evaluate
        model_dir = os.path.dirname(model_path)
        evaluate_output_dir = os.path.join(model_dir, f"evaluation/{game_objects['observation_model'].name}")
        os.makedirs(evaluate_output_dir, exist_ok=True)
        
        print(f"\nRunning evaluation and saving results to: {evaluate_output_dir}")
        evaluate(num_episodes=1000, single_strategy="DQNStrategy", output_path=evaluate_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent.")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to run (default: 2000)")
    parser.add_argument("--plot", action="store_true", help="Collect statistics and generate plots at the end")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training and save to models directory")
    args = parser.parse_args()

    main(episodes=args.episodes, plot_stats=args.plot, evaluate_after=args.evaluate)
