import os
from marl.agent import SharedAgent
from config.observation import ObservationModel

def create_agent(observation_model: ObservationModel, model_path=None, is_evaluation=False):
    observation_dim = observation_model.get_observation_dim()
    action_dim = observation_model.get_action_dim()

    if not model_path or not os.path.exists(model_path):
        model_path = None
        print("Model path does not exist.")

    agent = SharedAgent(observation_dim, action_dim, model_path=model_path)

    if is_evaluation:
        agent.policy_net.eval()

    return agent

def agent_based_strategy(observation_model: ObservationModel, agent: SharedAgent,  explore=True):
    def strategy(player, players):
        obs = observation_model.create_observation(player, players)
        action = agent.act(obs, explore)
        targets = observation_model.get_targets(player, players)
        return targets[action]
    return strategy