from abc import ABC, abstractmethod
from core.player import Player
from typing import List, Optional, Dict

class NullObservationModel:
    @property
    def name(self) -> str:
        """Name of the null observation model."""
        return "NullObservationModel"

    def initialize(self, players: list[Player]):
        """Null observation model does nothing."""
        pass
    def update(self, shots: list[tuple], remaining_rounds:  int | None = None):
        """Null observation model does nothing."""
        pass

class ObservationModel(NullObservationModel, ABC):
    """Base observation model with singleton pattern"""
    
    _instances = {}  # Dictionary to store instances by class name
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern: ensure only one instance per observation model class"""
        class_name = cls.__name__
        if class_name not in cls._instances:
            cls._instances[class_name] = super(ObservationModel, cls).__new__(cls)
        return cls._instances[class_name]
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the observation model."""
        pass

    @abstractmethod
    def create_observation(self, player: Player, players: list[Player]): pass

    @abstractmethod
    def get_observation_dim(self): pass

    @abstractmethod
    def get_action_dim(self): pass

    @abstractmethod
    def get_targets(self, player: Player, players: list[Player]): pass

    # New robust action-to-target mapping methods
    def get_action_space_mapping(self, player: Player, players: List[Player]) -> Dict[int, Player]:
        """
        Return a mapping from action index to target player.
        This provides a complete view of the action space for debugging.
        
        Returns:
            Dict mapping action_index -> target_player
        """
        targets = self.get_targets(player, players)
        return {i: target for i, target in enumerate(targets)}
    
    def get_target_from_action(self, player: Player, players: List[Player], action_index: int) -> Optional[Player]:
        """
        Convert action index to target player with validation.
        
        Args:
            player: The acting player
            players: All players in the game
            action_index: The action index from the agent
            
        Returns:
            Target player if valid, None if invalid action
        """
        try:
            targets = self.get_targets(player, players)
            if action_index < 0 or action_index >= len(targets):
                return None
            return targets[action_index]
        except (IndexError, TypeError):
            return None
    
    def get_action_from_target(self, player: Player, players: List[Player], target: Player) -> Optional[int]:
        """
        Convert target player to action index.
        
        Args:
            player: The acting player
            players: All players in the game
            target: The target player
            
        Returns:
            Action index if target is valid, None if target not in action space
        """
        try:
            targets = self.get_targets(player, players)
            return targets.index(target)
        except (ValueError, TypeError):
            return None
    
    def validate_action(self, player: Player, players: List[Player], action_index: int) -> bool:
        """
        Validate that an action index is valid for the current game state.
        
        Returns:
            True if action is valid, False otherwise
        """
        target = self.get_target_from_action(player, players, action_index)
        return target is not None
    
    def reset(self):
        """Reset observation model state. Override in subclasses if needed."""
        pass
    
    @classmethod
    def clear_instances(cls):
        """Clear all singleton instances - useful for testing or when switching configurations"""
        cls._instances.clear()