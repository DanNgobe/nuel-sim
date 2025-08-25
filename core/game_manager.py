# core/game_manager.py
import math
import random
from typing import List, Dict, Any, Tuple, Optional
from .game import Game
from .player import Player
from .gameplay import GamePlay
from .observation.observation_model import ObservationModel
from .strategies import BaseStrategy, TargetRandom

class GameManager:
    """
    Manages multiple game instances and serves as a training environment
    Singleton pattern implementation
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GameManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, 
                 num_players: int = 3,
                 gameplay: GamePlay = None,
                 observation_model: ObservationModel = None,
                 max_rounds: int = None,
                 marksmanship_range: Tuple[float, float] = (0.3, 0.9),
                 strategies: List[BaseStrategy] = None,
                 has_ghost: bool = False,
                 screen_width: int = 800,
                 screen_height: int = 600):
        
        # Only initialize once
        if GameManager._initialized:
            return
            
        self.num_players = num_players
        self.gameplay = gameplay
        self.observation_model = observation_model
        self.max_rounds = max_rounds
        self.marksmanship_range = marksmanship_range
        self.strategies = strategies or []
        self.has_ghost = has_ghost
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Game state
        self.current_game = None
        
        # For RL training
        self.prev_observations = {}
        self.prev_alive_state = {}
        
        GameManager._initialized = True
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (useful for testing)"""
        cls._instance = None
        cls._initialized = False
    
    def create_players(self) -> List[Player]:
        """Create players with positions in a circular formation"""
        center_x, center_y, radius = self.screen_width // 2, self.screen_height // 2, 200
        players = []
        
        # Create regular players
        for i in range(self.num_players):
            angle = 2 * math.pi * i / self.num_players
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Get strategy for this player
            strategy = self.strategies[i] if i < len(self.strategies) else TargetRandom()
            
            # Generate accuracy
            accuracy = random.uniform(*self.marksmanship_range)
            
            players.append(Player(
                id=i, 
                name=f"P{i+1}", 
                accuracy=accuracy, 
                x=x, 
                y=y, 
                strategy=strategy
            ))
        
        # Create ghost player if needed
        if self.has_ghost:
            players.append(Player(
                id=len(players),
                name="Ghost",
                accuracy=-1,
                x=center_x,
                y=center_y,
                alive=False
            ))
        
        return players
    
    def reset_game(self) -> Game:
        """Create a new game instance"""
        players = self.create_players()
        self.current_game = Game(
            players=players,
            gameplay=self.gameplay,
            observation_model=self.observation_model,
            max_rounds=self.max_rounds
        )
        
        # Reset previous observations for RL
        self.prev_observations = {
            player.id: self.observation_model.create_observation(player, players) 
            for player in players
        }
        
        # Store initial alive state
        self.prev_alive_state = {
            player.id: player.alive for player in players
        }
        
        return self.current_game
    
    def step(self) -> Tuple[Dict, Dict, bool, Dict]:
        """
        Advance the game by one turn
        Returns: observations, rewards, done, info
        """
        if self.current_game is None:
            self.reset_game()
        
        # Store previous observations for all players
        current_players = self.current_game.players
        self.prev_observations = {
            player.id: self.observation_model.create_observation(player, current_players) 
            for player in current_players
        }
        
        # Store previous alive state before the turn
        self.prev_alive_state = {
            player.id: player.alive for player in current_players
        }
        
        # If game is over, reset
        if self.current_game.is_over():
            rewards = self._calculate_rewards()
            self.reset_game()
            return self._get_observations(), rewards, True, self._get_info()
        
        # Run a turn
        self.current_game.run_turn()
        
        # Get game state information
        observations = self._get_observations()
        rewards = self._calculate_rewards()
        done = self.current_game.is_over()
        info = self._get_info()
        
        return observations, rewards, done, info
    
    def _get_observations(self) -> Dict[str, Any]:
        """Get current observations for all players for RL training"""
        if not self.current_game or not self.observation_model:
            return {}
        
        observations = {}
        for player in self.current_game.players:
            if player.alive:
                observations[player.id] = self.observation_model.create_observation(
                    player, self.current_game.players
                )
        return observations
    
    def _calculate_rewards(self) -> Dict[str, float]:
        """Calculate rewards for all players based on the last turn"""
        rewards = {player.id: 0.0 for player in self.current_game.players}
        
        if not self.current_game or not self.current_game.history:
            return rewards
        
        # Get the last turn's history
        last_history = self.current_game.history[-1]
        
        for shooter, target, hit in last_history:
            if not shooter or not target:
                continue
                
            reward = 0.0
            
            # Penalty for shooting players that were already dead (except ghost)
            # Use previous alive state to check if target was alive before the shot
            target_was_alive = self.prev_alive_state.get(target.id, True)
            if not target_was_alive and target.name != "Ghost":
                reward -= 1.0
            
            # Survival bonus for alive players
            if shooter.alive:
                reward += 0.1
            
            # Game over rewards
            if self.current_game.is_over():
                alive_players = [p for p in self.current_game.players if p.alive]
                if shooter.alive and len(alive_players) > 0:
                    # Divide winning bonus among alive players
                    reward += self.num_players / len(alive_players)
            
            rewards[shooter.id] = reward
        
        return rewards
    
    def _get_info(self) -> Dict:
        """Get additional info about the game state"""
        if not self.current_game:
            return {}
        
        return {
            "round": self.current_game.round_number,
            "alive_players": len(self.current_game.get_alive_players()),
            "history": self.current_game.history[-1] if self.current_game.history else []
        }
    
    def get_prev_observations(self) -> Dict[str, Any]:
        """Get previous observations for RL training"""
        return self.prev_observations.copy()
    
    def get_prev_alive_state(self) -> Dict[int, bool]:
        """Get previous alive state for RL training"""
        return self.prev_alive_state.copy()
    
    def get_targets_for_player(self, player: Player) -> List[Player]:
        """Get valid targets for a specific player"""
        if not self.current_game:
            return []
        return self.observation_model.get_targets(player, self.current_game.players)
    
    def run_episode(self) -> Dict:
        """Run a complete game episode"""
        self.reset_game()
        info = []
        while not self.current_game.is_over():
            info.append(self.step())
        return info