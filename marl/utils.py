import os
from marl.settings import OBSERVATION_MODEL
from marl.agent import SharedAgent

def create_agent(num_players, model_path=None, is_evaluation=False):
    """Creates a shared agent using the observation model."""
    observation_dim = OBSERVATION_MODEL.get_observation_dim(num_players)
    action_dim = OBSERVATION_MODEL.get_action_dim(num_players)

    if not model_path or not os.path.exists(model_path):
        model_path = None
        print("Model path does not exist.")

    agent = SharedAgent(observation_dim, action_dim, model_path=model_path)

    if is_evaluation:
        agent.policy_net.eval()

    return agent

def agent_based_strategy(agent: SharedAgent, explore=True):
    """Returns a function that uses the agent to pick targets deterministically."""
    def strategy(player, players):
        obs = OBSERVATION_MODEL.create_observation(player, players)
        action = agent.act(obs, explore)
        others = [pl for pl in players if pl != player]
        return others[action]
    return strategy