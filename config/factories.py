# config/factories.py
# Factory functions to create objects from string configurations
# This module breaks the circular import by doing the imports only when needed
import os

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
        SortedObservation, 
        BayesianMeanObservation,
        BayesianAbstentionObservation,
        TurnAwareThreatObservation,
        SimpleObservation,
        NoInfoObservation
    )
    
    model_classes = {
        "SortedObservation": SortedObservation,
        "BayesianMeanObservation": BayesianMeanObservation,
        "BayesianAbstentionObservation": BayesianAbstentionObservation,
        "TurnAwareThreatObservation": TurnAwareThreatObservation,
        "SimpleObservation": SimpleObservation,
        "NoInfoObservation": NoInfoObservation,
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown observation model type: {model_type}")
    
    return model_classes[model_type](**params)

def create_strategy(strategy_type: str, observation_model=None, model_path=None):
    """Factory function to create strategy objects from string identifiers"""
    from core.strategies import (
        TargetStrongest, 
        TargetWeaker, 
        TargetStronger, 
        TargetRandom, 
        TargetNearest,
        TargetBelievedStrongest
    )
    
    if strategy_type == "RLStrategy":
        from rllib_marl.strategy import RLlibStrategy
        from . import config_loader
        
        config = config_loader.get_config()
        
        if observation_model is None:
            observation_model = create_observation_model(
                config['observation']['model_type'],
                config['observation']['params']
            )
        
        # Use provided model_path or build default checkpoint path
        if model_path:
            checkpoint_path = os.path.abspath(model_path)
            print(f"Using model from: {checkpoint_path}")
        else:
            # Build checkpoint path from config
            algorithm = config['rllib']['algorithm']
            base_path = config['rllib']['checkpoint_base_path']
            num_players = config['game']['num_players']
            gameplay_type = config['gameplay']['type']
            obs_model_type = config['observation']['model_type']
            checkpoint_path = os.path.abspath(f"{base_path}/{algorithm}/{num_players}_players/{gameplay_type}_{obs_model_type}")
        
        return RLlibStrategy(
            checkpoint_path=checkpoint_path,
            observation_model=observation_model
        )

    if strategy_type == "TargetBelievedStrongest":
        if observation_model is None:
            from . import config_loader
            config = config_loader.get_config()
            observation_model = create_observation_model(
                config['observation']['model_type'],
                config['observation']['params']
            )
        return TargetBelievedStrongest(observation_model)

    strategy_classes = {
        "TargetStrongest": TargetStrongest,
        "TargetWeaker": TargetWeaker,
        "TargetStronger": TargetStronger,
        "TargetRandom": TargetRandom,
        "TargetNearest": TargetNearest,
    }
    
    if strategy_type not in strategy_classes:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    return strategy_classes[strategy_type]()


def create_strategies_list(strategy_types: list, observation_model=None, use_random_for_training=False, model_path=None):
    """Factory function to create a list of strategy objects"""
    if use_random_for_training:
        # Replace RLStrategy with TargetRandom for training
        safe_strategy_types = ["TargetRandom" if s == "RLStrategy" else s for s in strategy_types]
        return [create_strategy(strategy_type, observation_model, model_path) for strategy_type in safe_strategy_types]
    return [create_strategy(strategy_type, observation_model, model_path) for strategy_type in strategy_types]

# Convenience function to get all configured objects at once
def create_game_objects(config_path=None, use_random_for_training=False, model_path=None):
    from . import config_loader
    
    if config_path:
        config_loader.load_config(config_path)
    
    config = config_loader.get_config()
    
    gameplay = create_gameplay(config['gameplay']['type'])
    observation_model = create_observation_model(
        config['observation']['model_type'], 
        config['observation']['params']
    )
    strategies = create_strategies_list(config['players']['strategy_types'], observation_model, use_random_for_training, model_path)
    
    return {
        'gameplay': gameplay,
        'observation_model': observation_model,
        'strategies': strategies
    }
