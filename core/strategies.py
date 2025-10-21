# core/strategies.py
from abc import ABC, abstractmethod
import random
import math
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .player import Player

class BaseStrategy(ABC):
    """Abstract base class for all strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def choose_target(self, me: "Player", players: List["Player"], observation=None) -> tuple[Optional["Player"], Optional[int]]:
        """
        Choose a target from the list of players
        Returns: (target_player, action_index)
        """
        pass
    
    def __call__(self, me: "Player", players: List["Player"], observation=None) -> tuple[Optional["Player"], Optional[int]]:
        return self.choose_target(me, players, observation)
    
    def __str__(self) -> str:
        return self.name

class TargetStrongest(BaseStrategy):
    def __init__(self):
        super().__init__("target_strongest")
    
    def choose_target(self, me: "Player", players: List["Player"], observation=None) -> tuple[Optional["Player"], Optional[int]]:
        alive = [p for p in players if p != me and p.alive]
        if not alive:
            return None, None
        
        target = max(alive, key=lambda p: p.accuracy)
        # Find action index (position in players list excluding self)
        others = [p for p in players if p != me]
        action_index = others.index(target) if target in others else None
        return target, action_index

class TargetWeaker(BaseStrategy):
    def __init__(self):
        super().__init__("target_weaker")
    
    def choose_target(self, me: "Player", players: List["Player"], observation=None) -> tuple[Optional["Player"], Optional[int]]:
        alive = [p for p in players if p != me and p.alive]
        if not alive:
            return None, None
        
        target = min(alive, key=lambda p: p.accuracy)
        # Find action index (position in players list excluding self)
        others = [p for p in players if p != me]
        action_index = others.index(target) if target in others else None
        return target, action_index

class TargetStronger(BaseStrategy):
    def __init__(self):
        super().__init__("target_stronger")
    
    def choose_target(self, me: "Player", players: List["Player"], observation=None) -> tuple[Optional["Player"], Optional[int]]:
        alive = [p for p in players if p != me and p.alive]
        if not alive:
            return None, None
        
        target = max(alive, key=lambda p: p.accuracy - me.accuracy)
        # Find action index (position in players list excluding self)
        others = [p for p in players if p != me]
        action_index = others.index(target) if target in others else None
        return target, action_index

class TargetRandom(BaseStrategy):
    def __init__(self):
        super().__init__("target_random")
    
    def choose_target(self, me: "Player", players: List["Player"], observation=None) -> tuple[Optional["Player"], Optional[int]]:
        alive = [p for p in players if p != me and p.alive]
        if not alive:
            return None, None
        
        target = random.choice(alive)
        # Find action index (position in players list excluding self)
        others = [p for p in players if p != me]
        action_index = others.index(target) if target in others else None
        return target, action_index

class TargetNearest(BaseStrategy):
    def __init__(self):
        super().__init__("target_nearest")
    
    def choose_target(self, me: "Player", players: List["Player"], observation=None) -> tuple[Optional["Player"], Optional[int]]:
        alive = [p for p in players if p != me and p.alive]
        if not alive:
            return None, None
        
        def distance(p1: "Player", p2: "Player") -> float:
            return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
        target = min(alive, key=lambda p: distance(me, p))
        # Find action index (position in players list excluding self)
        others = [p for p in players if p != me]
        action_index = others.index(target) if target in others else None
        return target, action_index

class TargetBelievedStrongest(BaseStrategy):
    def __init__(self, observation_model):
        super().__init__("target_believed_strongest")
        self.observation_model = observation_model
    
    def choose_target(self, me: "Player", players: List["Player"], observation=None) -> tuple[Optional["Player"], Optional[int]]:
        alive = [p for p in players if p != me and p.alive]
        if not alive:
            return None, None
        
        # Check if observation model is Bayesian
        if "Bayesian" in self.observation_model.name:
            # Use believed strength from observation model
            target = max(alive, key=lambda p: self.observation_model.global_beliefs[p.id]['mean'])
        else:
            # Use actual accuracy
            target = max(alive, key=lambda p: p.accuracy)
        
        # Find action index
        others = [p for p in players if p != me]
        action_index = others.index(target) if target in others else None
        return target, action_index
