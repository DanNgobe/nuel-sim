# marl/train.py
from core.game_manager import GameManager
from ddqn_marl.replay_buffer import ReplayBuffer
from ddqn_marl.strategy import DDQNStrategy
from ddqn_marl.settings import MEMORY_SIZE, TARGET_UPDATE_FREQ, get_model_path
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
    
    # Create DDQN strategy
    ddqn_strategy = DDQNStrategy(
        config.OBSERVATION_MODEL, 
        model_path=model_path, 
        explore=True
    )
    
    # Create strategies list (all players use DDQN strategy)
    strategies = [ddqn_strategy for _ in range(config.NUM_PLAYERS)]
    
    # Create GameManager with DDQN strategies
    game_manager = GameManager(
        num_players=config.NUM_PLAYERS,
        gameplay=config.GAME_PLAY,
        observation_model=config.OBSERVATION_MODEL,
        max_rounds=config.NUM_ROUNDS,
        marksmanship_range=config.MARKSMANSHIP_RANGE,
        strategies=strategies,
        has_ghost=config.HAS_GHOST,
        screen_width=config.SCREEN_WIDTH,
        screen_height=config.SCREEN_HEIGHT
    )
    
    replay_buffer = ReplayBuffer(MEMORY_SIZE)
    agent = ddqn_strategy.get_agent()

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
                    shooter_obs = shooter.prev_observation
                    reward = rewards.get(shooter.id, 0.0)
    
                    next_obs = observations.get(shooter.id, shooter_obs)
                    replay_buffer.push(shooter_obs, shooter.last_action, reward, next_obs, done)
            
            # Update agent
            agent.update(replay_buffer)

        # Decay epsilon
        agent.decay_epsilon(episode)
        
        # Update target network periodically
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # Progress printout
        if episode % 100 == 0:
            print(f"Episode {episode}: Epsilon={agent.epsilon:.3f}")

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model after training
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"Training completed and model saved to {model_path}.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate game performance.")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to run (default: 2000)")
    args = parser.parse_args()

    main(episodes=args.episodes)

    
