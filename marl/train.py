# marl/train.py
from core import Game, Player
from marl.replay_buffer import ReplayBuffer
from marl.utils import create_agent, agent_based_strategy
from marl.settings import MEMORY_SIZE, TARGET_UPDATE_FREQ
import config
import torch
import random
import argparse
import os

def main(episodes=2000):
    num_players = config.NUM_PLAYERS
    agent = create_agent(config.OBSERVATION_MODEL, model_path=config.MODEL_PATH, is_evaluation=False)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)


    for episode in range(episodes):
        # Create fresh players each episode
        players = []
        for i in range(num_players):
            accuracy = random.uniform(*config.MARKSMANSHIP_RANGE)  # players have different accuracies
            players.append(Player(id=i, name=f"P{i}", accuracy=accuracy, strategy=agent_based_strategy(config.OBSERVATION_MODEL,agent, explore=True)))
        
        if (config.HAS_GHOST): # Create a ghost player
            players.append(Player(id=num_players, name="Ghost", accuracy=-1, alive=False))
       
        game = Game(players, gameplay=config.GAME_PLAY, observation_model=config.OBSERVATION_MODEL, max_rounds=config.NUM_ROUNDS)

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
                        if target.name != "Ghost": # Shooting the ghost is not a problem
                            reward -= 1.0 # punish for shooting dead players
                     
                    # if hit: # This is bad, agent could learn to gang up on one weak player just to get the reward
                    #     reward += 1.0  # hit bonus
                    
                    if shooter.alive: # careful when there is abstention(players might exploit it)
                        reward += 0.1 # survival bonus

                    if done:
                        # Divide the winning bonus among all alive players
                        alive_players = [p for p in players if p.alive]
                        if shooter.alive and len(alive_players) > 0:
                            reward += config.NUM_PLAYERS / len(alive_players) # Abstention can be seen as a strategy when the reward structure favors it

                    next_obs = config.OBSERVATION_MODEL.create_observation(shooter, players)
                    targets = config.OBSERVATION_MODEL.get_targets(shooter, players)
                    action = targets.index(target)

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
    torch.save(agent.policy_net.state_dict(), config.MODEL_PATH)
    print("Training completed and model saved.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate game performance.")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to run (default: 2000)")
    args = parser.parse_args()

    main(episodes=args.episodes)

    
