# dqn_marl/train.py
from core.game_manager import GameManager
from dqn_marl.replay_buffer import ReplayBuffer
from dqn_marl.strategy import DQNStrategy
from dqn_marl.settings import MEMORY_SIZE, get_model_path
import config
import torch
import argparse
import os

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
        # Reset game for new episode
        game_manager.reset_game()
        
        done = False
        while not done:
            # Run one step
            observations, rewards, done, info = game_manager.step()
            
            # Process last turn's history for experience replay
            if 'history' in info and info['history']:
                for shooter, target, hit in info['history']:
                    if shooter.prev_observation is not None and shooter.last_action is not None:
                        shooter_obs = shooter.prev_observation
                        reward = rewards.get(shooter.id, 0.0)
        
                        next_obs = observations.get(shooter.id, shooter_obs)
                        replay_buffer.push(shooter_obs, shooter.last_action, reward, next_obs, done)
            
            # Update agent
            agent.update(replay_buffer)

        # Decay epsilon
        agent.decay_epsilon(episode)

        # Progress printout
        if episode % 100 == 0:
            print(f"Episode {episode}: Epsilon={agent.epsilon:.3f}")

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
