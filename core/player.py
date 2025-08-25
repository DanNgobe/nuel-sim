import random

class Player:
    def __init__(self, id, name, accuracy, x=0, y=0, strategy=None, alive=True, observation_model=None):
        self.id = id
        self.name = name
        self.accuracy = accuracy
        self.x = x
        self.y = y
        self.alive = alive
        self.observation_model = observation_model
        
        # State tracking for RL/strategies
        self.prev_observation = None
        self.last_action = None
        
        # Import strategy here to avoid circular import
        if strategy is None:
            from .strategies import TargetRandom
            strategy = TargetRandom()
        self.strategy = strategy

    def choose_target(self, players):
        # Create observation if observation model is available
        observation = None
        if self.observation_model:
            observation = self.observation_model.create_observation(self, players)
            self.prev_observation = observation
        
        chosen_player, action = self.strategy.choose_target(self, players, observation)
        self.last_action = action

        return chosen_player

    def shoot(self, target):
        if not self.alive or not target:
            raise ValueError("Player is not alive or target is None")
        return random.random() < self.accuracy