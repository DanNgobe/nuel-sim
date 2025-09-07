# dqn_marl/train.py
from core.game_manager import GameManager
from dqn_marl.replay_buffer import ReplayBuffer
from dqn_marl.strategy import DQNStrategy
from dqn_marl.settings import MEMORY_SIZE, get_model_path, BATCH_SIZE
import config
import torch
import argparse
import os
import numpy as np

def main(episodes=2000):
    # Get model path
    model_path = get_model_path(
        config.NUM_PLAYERS, 
        config.GAME_PLAY.__class__.__name__, 
        config.OBSERVATION_MODEL.name
    )
    
    # Create DQN strategy
    dqn_strategy = DQNStrategy(
        config.OBSERVATION_MODEL, 
        model_path=model_path, 
        explore=True
    )
    
    # Create strategies list (all players use DQN strategy)
    strategies = [dqn_strategy for _ in range(config.NUM_PLAYERS)]
    
    # Create GameManager with DQN strategies
    game_manager = GameManager(
        num_players=config.NUM_PLAYERS,
        gameplay=config.GAME_PLAY,
        observation_model=config.OBSERVATION_MODEL,
        max_rounds=config.NUM_ROUNDS,
        marksmanship_range=config.MARKSMANSHIP_RANGE,
        strategies=strategies,
        assigned_accuracies=config.ASSIGNED_ACCURACIES,
        has_ghost=config.HAS_GHOST,
        screen_width=config.SCREEN_WIDTH,
        screen_height=config.SCREEN_HEIGHT
    )
    
    replay_buffer = ReplayBuffer(MEMORY_SIZE)
    agent = dqn_strategy.get_agent()

    for episode in range(episodes):
        # Reset environment for new episode (Gymnasium interface)
        observations, infos = game_manager.reset()
        prev_observations = observations.copy()
        
        episode_reward = 0
        steps = 0
        terminated = False
        
        while not terminated:
            # Get actions from each player using current observations
            actions = {}
            for player_id in game_manager.agents:
                if player_id in observations:
                    # Get the player object
                    player = next(p for p in game_manager.current_game.players if p.id == player_id)
                    
                    # Only get actions for alive players
                    if player.alive:
                        obs = observations[player_id]
                        action_idx = agent.act(obs, explore=True)
                        
                        # Convert action index to target player ID
                        targets = config.OBSERVATION_MODEL.get_targets(player, game_manager.current_game.players)
                        if action_idx < len(targets):
                            target = targets[action_idx]
                            actions[player_id] = target.id
                        else:
                            # Invalid action, choose random target
                            alive_targets = [p for p in targets if p.alive]
                            if alive_targets:
                                actions[player_id] = alive_targets[0].id
            
            # Skip if no actions to take
            if not actions:
                break
                
            # Execute step with actions
            observations, rewards, terminateds, truncateds, infos = game_manager.step(actions)
            
            # Check if episode is terminated
            terminated = terminateds.get("__all__", False) or truncateds.get("__all__", False)
            
            # Store experiences for all players that took actions
            for player_id in actions:
                if player_id in prev_observations and player_id in rewards:
                    player = next(p for p in game_manager.current_game.players if p.id == player_id)
                    prev_obs = prev_observations[player_id]
                    
                    # Convert target ID back to action index for experience replay
                    target_id = actions[player_id]
                    targets = config.OBSERVATION_MODEL.get_targets(player, game_manager.current_game.players)
                    action_idx = None
                    for i, target in enumerate(targets):
                        if target.id == target_id:
                            action_idx = i
                            break
                    
                    if action_idx is not None:
                        reward = rewards[player_id]
                        next_obs = observations.get(player_id, prev_obs)
                        done = terminateds.get(player_id, False)
                        
                        replay_buffer.push(prev_obs, action_idx, reward, next_obs, done)
                        episode_reward += reward
            
            # Update agent if we have enough samples
            if len(replay_buffer) >= BATCH_SIZE:
                agent.update(replay_buffer)
            
            # Update previous observations for next iteration
            prev_observations = observations.copy()
            steps += 1
            
            # Prevent infinite loops
            if steps > 1000:
                break

        # Decay epsilon
        agent.decay_epsilon(episode, total_episodes=episodes)

        # Progress printout
        if episode % 100 == 0 or episode < 10:
            alive_count = sum(1 for player_id in game_manager.agents if infos.get(player_id, {}).get("alive", False))
            print(f"Episode {episode}: Epsilon={agent.epsilon:.3f}, Steps: {steps}, Total Reward: {episode_reward:.2f}, Alive players: {alive_count}")

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model after training
    agent.save_model(model_path)
    print(f"Training completed and model saved to {model_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent.")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to run (default: 2000)")
    args = parser.parse_args()

    main(episodes=args.episodes)
