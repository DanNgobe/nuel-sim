# config/factories.py
# Factory functions to create objects from string configurations
# This module breaks the circular import by doing the imports only when needed

def create_gameplay(gameplay_type: str):
    """Factory function to create gameplay objects from string identifiers"""
    from core.gameplay import SequentialGamePlay, RandomGamePlay, SimultaneousGamePlay
    
    gameplay_classes = {
        "SequentialGamePlay": SequentialGamePlay,
        "RandomGamePlay": RandomGamePlay, 
        "SimultaneousGamePlay": SimultaneousGamePlay,
    }
    
    if gameplay_type not in gameplay_classes:
        raise ValueError(f"Unknown gameplay type: {gameplay_type}")
    
    return gameplay_classes[gameplay_type]()


def create_observation_model(model_type: str, params: dict):
    """Factory function to create observation model objects from string identifiers"""
    from core.observation import (
        ThreatLevelObservation, 
        BayesianMeanObservation,
        TurnAwareThreatObservation
    )
    
    model_classes = {
        "ThreatLevelObservation": ThreatLevelObservation,
        "BayesianMeanObservation": BayesianMeanObservation,
        "TurnAwareThreatObservation": TurnAwareThreatObservation,
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown observation model type: {model_type}")
    
    return model_classes[model_type](**params)

def create_strategy(strategy_type: str, observation_model=None):
    """Factory function to create strategy objects from string identifiers"""
    from core.strategies import (
        TargetStrongest, 
        TargetWeaker, 
        TargetStronger, 
        TargetRandom, 
        TargetNearest
    )
    
    if strategy_type == "RLlibStrategy":
        from rllib_marl.strategy import RLlibStrategy
        from . import settings    
        
        # Use passed observation model or create new one if none provided
        if observation_model is None:
            observation_model = create_observation_model(
                settings.OBSERVATION_MODEL_TYPE, 
                settings.OBSERVATION_MODEL_PARAMS
            )
                
        return RLlibStrategy(
            checkpoint_path=settings.RLLIB_CHECKPOINT_PATH,
            observation_model=observation_model
        )
    
    if strategy_type == "DQNStrategy":
        from dqn_marl.strategy import DQNStrategy
        from . import settings    
        
        # Use passed observation model or create new one if none provided
        if observation_model is None:
            observation_model = create_observation_model(
                settings.OBSERVATION_MODEL_TYPE, 
                settings.OBSERVATION_MODEL_PARAMS
            )
        
        model_path = get_model_path(
            settings.NUM_PLAYERS,
            settings.GAME_PLAY_TYPE, 
            settings.OBSERVATION_MODEL_TYPE,
            settings.OBSERVATION_MODEL_PARAMS
        )
                
        return DQNStrategy(
            observation_model=observation_model,
            model_path=model_path,
            explore=False  # Default to no exploration for evaluation
        )
    
    if strategy_type == "PPOStrategy":
        from ppo_marl.strategy import PPOStrategy
        from . import settings    
        
        # Use passed observation model or create new one if none provided
        if observation_model is None:
            observation_model = create_observation_model(
                settings.OBSERVATION_MODEL_TYPE, 
                settings.OBSERVATION_MODEL_PARAMS
            )
        
        model_path = get_ppo_model_path(
            settings.NUM_PLAYERS,
            settings.GAME_PLAY_TYPE, 
            settings.OBSERVATION_MODEL_TYPE,
            settings.OBSERVATION_MODEL_PARAMS
        )
                
        return PPOStrategy(
            observation_model=observation_model,
            model_path=model_path,
            explore=False  # Default to no exploration for evaluation
        )

    strategy_classes = {
        "TargetStrongest": TargetStrongest,
        "TargetWeaker": TargetWeaker,
        "TargetStronger": TargetStronger,
        "TargetRandom": TargetRandom,
        "TargetNearest": TargetNearest,
        "PPOStrategy": None,  # Handled above
    }
    
    if strategy_type not in strategy_classes:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    return strategy_classes[strategy_type]()


def create_strategies_list(strategy_types: list, observation_model=None):
    """Factory function to create a list of strategy objects"""
    return [create_strategy(strategy_type, observation_model) for strategy_type in strategy_types]


def get_model_path(num_players: int, gameplay_type: str, observation_model_type: str, observation_params: dict):
    """Generate model path based on configuration"""
    # Create a temporary observation model to get its name
    obs_model = create_observation_model(observation_model_type, observation_params)
    return f"dqn_marl/models/{num_players}_{gameplay_type}_{obs_model.name}.pth"

def get_ppo_model_path(num_players: int, gameplay_type: str, observation_model_type: str, observation_params: dict):
    """Generate PPO model path based on configuration"""
    # Create a temporary observation model to get its name
    obs_model = create_observation_model(observation_model_type, observation_params)
    return f"ppo_marl/models/{num_players}_{gameplay_type}_{obs_model.name}.pth"


# Convenience function to get all configured objects at once
def create_game_objects():
    """Create all game objects based on current configuration"""
    from . import settings
    
    gameplay = create_gameplay(settings.GAME_PLAY_TYPE)
    observation_model = create_observation_model(
        settings.OBSERVATION_MODEL_TYPE, 
        settings.OBSERVATION_MODEL_PARAMS
    )
    strategies = create_strategies_list(settings.ASSIGNED_STRATEGY_TYPES, observation_model)
    model_path = get_model_path(
        settings.NUM_PLAYERS,
        settings.GAME_PLAY_TYPE, 
        settings.OBSERVATION_MODEL_TYPE,
        settings.OBSERVATION_MODEL_PARAMS
    )
    
    return {
        'gameplay': gameplay,
        'observation_model': observation_model,
        'strategies': strategies,
        'model_path': model_path
    }
