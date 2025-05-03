# marl/train.py
from core import Game, Player
from marl.replay_buffer import ReplayBuffer
from marl.strategy import create_agent, agent_based_strategy, create_observation
import marl.settings as settings
import config
import torch
import random

def main():
    num_players = config.NUM_PLAYERS
    agent = create_agent(num_players, model_path=config.get_model_path(num_players, game_play=config.GAME_PLAY))
    replay_buffer = ReplayBuffer(settings.MEMORY_SIZE)

    episodes = settings.NUM_EPISODES

    for episode in range(episodes):
        # Create fresh players each episode
        players = []
        for i in range(num_players):
            accuracy = random.uniform(*config.MARKSMANSHIP_RANGE)  # players have different accuracies
            players.append(Player(f"P{i}", accuracy=accuracy, strategy=agent_based_strategy(agent)))

        game = Game(players, gameplay=config.GAME_PLAY)

        done = False
        while not done:
            prev_obs = {player.name: [player.alive, create_observation(player, players)] for player in players} # store observations for all players
            game.run_turn()
            done = game.is_over()
            last_history = game.history[-1] # last turn's history [(shooter, target, hit), ...]

            for shooter, target, hit in last_history:
                if shooter and target:
                    shooter_obs = prev_obs[shooter.name][1]
                    reward = 0.0
                    
                    if not prev_obs[target.name][0]:
                        reward -= 1.0 # punish for shooting dead players
                    
                    if shooter.name == target.name:
                        reward -= 1.0 # punish for shooting oneself
                    
                    if hit:
                        reward += 1.0  # hit bonus
                    
                    if shooter.alive:
                        reward += 0.1 # survival bonus

                    if done and shooter.alive:
                        reward += 5.0  # winning bonus

                    next_obs = create_observation(shooter, players)
                    action = players.index(target)

                    replay_buffer.push(shooter_obs, action, reward, next_obs, done)
            agent.update(replay_buffer)

        # Update target network periodically
        if episode % settings.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        # Small printout
        if episode % 100 == 0:
            print(f"Episode {episode}: Epsilon={agent.epsilon:.3f}")

    # Save model after training
    torch.save(agent.policy_net.state_dict(), config.get_model_path(num_players, game_play=config.GAME_PLAY))
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()
