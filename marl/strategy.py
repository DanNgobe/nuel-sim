#marl/strategy.py
import marl.settings as settings
import config as config
from marl.agent import SharedAgent
import torch

OBS_DIM = config.NUM_PLAYERS + 1 + (config.NUM_PLAYERS - 1) * 2  # one-hot + accuracy + others' alive/accuracy
ACTION_DIM = config.NUM_PLAYERS  # one action per player

def create_observation(player, players):
    """Observation includes:
    - Own index (one-hot encoded)
    - Own accuracy
    - Other players: [alive status, accuracy] * (N-1)
    """
    obs = []

    # Own index as one-hot vector
    index = players.index(player)
    index_one_hot = [1.0 if i == index else 0.0 for i in range(len(players))]
    obs.extend(index_one_hot)

    # Own accuracy
    obs.append(player.accuracy)

    # Others' alive/accuracy
    for p in players:
        if p != player:
            obs.append(1.0 if p.alive else 0.0)
            obs.append(p.accuracy)
    
    return obs

def agent_based_strategy(agent, explore=True):
    """Returns a function that uses the agent to pick targets deterministically."""
    def strategy(player, players):
        obs = create_observation(player, players)
        action = agent.act(obs, explore)
        # Choose among players
        action = min(action, len(players) - 1)  # safety
        return players[action]
    return strategy

def agent_strategy():
    agent = SharedAgent(observation_dim=OBS_DIM, action_dim=ACTION_DIM)
    agent.policy_net.load_state_dict(torch.load(f"marl/models/policy_net_{config.GAME_PLAY.__class__.__name__}_{config.NUM_PLAYERS}.pth", map_location=settings.DEVICE))
    agent.policy_net.eval()  # evaluation mode (no dropout, no batchnorm updates)

    return agent_based_strategy(agent, explore=False)