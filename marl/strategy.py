#marl/strategy.py
from marl.agent import SharedAgent

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

def get_observation_dim(num_players):
    """Returns the dimension of the observation space."""
    return num_players + 1 + (num_players - 1) * 2  # one-hot + accuracy + others' alive/accuracy

def get_action_dim(num_players):
    """Returns the dimension of the action space."""
    return num_players  # one action per player

def agent_based_strategy(agent, explore=True):
    """Returns a function that uses the agent to pick targets deterministically."""
    def strategy(player, players):
        obs = create_observation(player, players)
        action = agent.act(obs, explore)
        # Choose among players
        action = min(action, len(players) - 1)  # safety
        return players[action]
    return strategy

def create_agent(num_players, model_path=None, is_evaluation=False):
    """Creates a shared agent with the given number of players."""
    observation_dim = get_observation_dim(num_players)
    action_dim = get_action_dim(num_players)
    agent = SharedAgent(observation_dim, action_dim, model_path=model_path)
    if is_evaluation:
        agent.policy_net.eval()
    return agent