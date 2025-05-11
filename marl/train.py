# marl/train.py
from core import Game, Player
from marl.replay_buffer import ReplayBuffer
from marl.utils import create_agent, agent_based_strategy
from marl.settings import MEMORY_SIZE, TARGET_UPDATE_FREQ
import config
import torch
import random
import argparse

def main(episodes=2000):
    num_players = config.NUM_PLAYERS
    agent = create_agent(config.OBSERVATION_MODEL, model_path=config.get_model_path(num_players, game_play=config.GAME_PLAY))
    replay_buffer = ReplayBuffer(MEMORY_SIZE)


    for episode in range(episodes):
        # Create fresh players each episode
        players = []
        for i in range(num_players):
            accuracy = random.uniform(*config.MARKSMANSHIP_RANGE)  # players have different accuracies
            players.append(Player(f"P{i}", accuracy=accuracy, strategy=agent_based_strategy(config.OBSERVATION_MODEL,agent, explore=True)))

        game = Game(players, gameplay=config.GAME_PLAY)

        done = False
        while not done:
            prev_obs = {player.name: [player.alive, config.OBSERVATION_MODEL.create_observation(player, players)] for player in players} # store observations for all players
            game.run_turn()
            done = game.is_over()
            last_history = game.history[-1] # last turn's history [(shooter, target, hit), ...]

            for shooter, target, hit in last_history:
                if shooter and target:
                    shooter_obs = prev_obs[shooter.name][1]
                    reward = 0.0
                    
                    if not prev_obs[target.name][0]:
                        reward -= 1.0 # punish for shooting dead players
                     
                    if hit:
                        reward += 1.0  # hit bonus
                    
                    if shooter.alive:
                        reward += 0.1 # survival bonus

                    if done and shooter.alive:
                        reward += 5.0  # winning bonus

                    next_obs = config.OBSERVATION_MODEL.create_observation(shooter, players)

                    others = [p for p in players if p != shooter]
                    action = others.index(target)

                    replay_buffer.push(shooter_obs, action, reward, next_obs, done)
            agent.update(replay_buffer)

        agent.decay_epsilon(episode)
        # Update target network periodically
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # Small printout
        if episode % 100 == 0:
            print(f"Episode {episode}: Epsilon={agent.epsilon:.3f}")

    # Save model after training
    torch.save(agent.policy_net.state_dict(), config.get_model_path(num_players, game_play=config.GAME_PLAY))
    print("Training completed and model saved.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate game performance.")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to run (default: 2000)")
    args = parser.parse_args()

    main(episodes=args.episodes)

    
