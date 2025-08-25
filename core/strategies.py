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
        self.dependencies = {}
    
    def add_dependency(self, key: str, dependency):
        """Add a dependency to this strategy"""
        self.dependencies[key] = dependency
    
    def get_dependency(self, key: str):
        """Get a dependency by key"""
        return self.dependencies.get(key)
    
    @abstractmethod
    def choose_target(self, me: "Player", players: List["Player"]) -> Optional["Player"]:
        """Choose a target from the list of players"""
        pass
    
    def __call__(self, me: "Player", players: List["Player"]) -> Optional["Player"]:
        return self.choose_target(me, players)
    
    def __str__(self) -> str:
        return self.name

class TargetStrongest(BaseStrategy):
    def __init__(self):
        super().__init__("target_strongest")
    
    def choose_target(self, me: "Player", players: List["Player"]) -> Optional["Player"]:
        alive = [p for p in players if p != me and p.alive]
        return max(alive, key=lambda p: p.accuracy, default=None)

class TargetWeaker(BaseStrategy):
    def __init__(self):
        super().__init__("target_weaker")
    
    def choose_target(self, me: "Player", players: List["Player"]) -> Optional["Player"]:
        alive = [p for p in players if p != me and p.alive]
        return min(alive, key=lambda p: p.accuracy, default=None)

class TargetStronger(BaseStrategy):
    def __init__(self):
        super().__init__("target_stronger")
    
    def choose_target(self, me: "Player", players: List["Player"]) -> Optional["Player"]:
        alive = [p for p in players if p != me and p.alive]
        return max(alive, key=lambda p: p.accuracy - me.accuracy, default=None)

class TargetRandom(BaseStrategy):
    def __init__(self):
        super().__init__("target_random")
    
    def choose_target(self, me: "Player", players: List["Player"]) -> Optional["Player"]:
        alive = [p for p in players if p != me and p.alive]
        return random.choice(alive) if alive else None

class TargetNearest(BaseStrategy):
    def __init__(self):
        super().__init__("target_nearest")
    
    def choose_target(self, me: "Player", players: List["Player"]) -> Optional["Player"]:
        alive = [p for p in players if p != me and p.alive]
        if not alive:
            return None
        
        def distance(p1: "Player", p2: "Player") -> float:
            return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
        return min(alive, key=lambda p: distance(me, p))
