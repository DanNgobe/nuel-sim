import math
import random
from typing import List, Dict, Any, Tuple, Optional, Set
import gymnasium as gym
import numpy as np
from core.game import Game
from core.player import Player
from core.gameplay import GamePlay
from core.observation.observation_model import ObservationModel, NullObservationModel
from core.strategies import BaseStrategy, TargetRandom
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from visual.pygame_visual import run_infinite_game_visual

class GameManager(MultiAgentEnv):
    """Multi-agent environment wrapper for the shooting game."""
    
    def __init__(self, 
                 num_players: int = 3,
                 gameplay: GamePlay = None,
                 observation_model: Optional[ObservationModel] = None,
                 max_rounds: Optional[int] = None,
                 marksmanship_range: Tuple[float, float] = (0.3, 0.9),
                 strategies: Optional[List[BaseStrategy]] = None,
                 assigned_accuracies: Optional[List[float]] = None,
                 has_ghost: bool = False,
                 screen_width: int = 800,
                 screen_height: int = 600):
        
        super().__init__()
        
        # Game parameters
        self.num_players = num_players
        self.gameplay = gameplay
        self.observation_model = observation_model or NullObservationModel()
        self.max_rounds = max_rounds
        self.marksmanship_range = marksmanship_range
        self.has_ghost = has_ghost
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.assigned_accuracies = assigned_accuracies or []
        
        # Initialize strategies
        if strategies is None:
            self.strategies = [TargetRandom() for _ in range(num_players)]
        else:
            self.strategies = strategies
            # If fewer strategies than players, fill with TargetRandom
            while len(self.strategies) < num_players:
                self.strategies.append(TargetRandom())
                    
        # Agent IDs
        self.agents = [i for i in range(num_players)]
        self.possible_agents = self.agents.copy()
        
        # Initialize observation and action spaces
        obs_dim = self.observation_model.get_observation_dim()
        action_dim = self.observation_model.get_action_dim()
        
        self.observation_spaces = {
            agent_id: gym.spaces.Box(
                low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
            for agent_id in self.agents
        }
        
        self.action_spaces = {
            agent_id: gym.spaces.Discrete(action_dim)
            for agent_id in self.agents
        }
        
        # Initialize game state
        self.current_game = None
        self.prev_alive_state = {}
        self.reset_game()

    def _create_players(self) -> List[Player]:
        """Create players with positions in a circular formation"""
        center_x, center_y = self.screen_width // 2, self.screen_height // 2
        radius = min(center_x, center_y) - 50 
        players = []
        
        # Generate unique accuracies for all players
        min_acc, max_acc = self.marksmanship_range
        step = 0.1
        available_accuracies = []
        
        # Create list of available accuracies in steps of 0.1
        current = min_acc
        while current <= max_acc:
            available_accuracies.append(round(current, 1))
            current += step
        
        # If we need more accuracies than available in range, extend the range
        while len(available_accuracies) < self.num_players:
            max_acc += step
            available_accuracies.append(round(max_acc, 1))
        
        # Shuffle to randomize assignment
        random.shuffle(available_accuracies)
        
        # Create regular players
        for i in range(self.num_players):
            angle = 2 * math.pi * i / self.num_players
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Get strategy for this player
            strategy = self.strategies[i] if i < len(self.strategies) else TargetRandom()
            
            # Use assigned accuracy if available, otherwise use unique accuracy from list
            if i < len(self.assigned_accuracies):
                accuracy = self.assigned_accuracies[i]
            else:
                accuracy = available_accuracies[i]
            
            players.append(Player(
                id=i, 
                name=f"P{i+1}", 
                accuracy=accuracy, 
                x=x, 
                y=y, 
                strategy=strategy,
                observation_model=self.observation_model
            ))
        
        # Create ghost player if needed
        if self.has_ghost:
            players.append(Player(
                id=len(players),
                name="Ghost",
                accuracy=-1,
                x=center_x,
                y=center_y,
                alive=False,
                observation_model=self.observation_model
            ))
        
        return players
    
    def reset_game(self) -> Game:
        """Create a new game instance."""
        players = self._create_players()
        self.current_game = Game(
            players=players,
            gameplay=self.gameplay,
            observation_model=self.observation_model,
            max_rounds=self.max_rounds
        )
        
        # Store initial alive state
        self.prev_alive_state = {
            player.id: player.alive for player in players
        }
        
        # Prepare first turn
        self.current_game.prepare_turn()
        
        return self.current_game

    def _calculate_rewards(self, shots: List[Tuple[Player, Player, bool]], game_over: bool) -> Dict[int, float]:
        """Calculate rewards for each player based on the shots taken."""
        rewards = {player.id: 0.0 for player in self.current_game.players}
        for shooter, target, hit in shots:
            if shooter.alive: # Survival reward
                rewards[shooter.id] += 1.0

            # Negative reward for shooting dead players (except ghosts)
            if target and not self.prev_alive_state.get(target.id, True) and target.name != "Ghost":
                rewards[shooter.id] -= 10.0
        
            if game_over:
                alive_players = self.current_game.get_alive_players()
                if shooter.alive and len(alive_players) > 0:
                    rewards[shooter.id] += 100 / len(alive_players)
        
        return rewards

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment and return initial observations."""
        super().reset(seed=seed, options=options)
        
        # Reset the game
        self.reset_game()
        
        # Get initial observations
        observations = {}
        infos = {}
        
        for player in self.current_game.players:
            agent_id = player.id
            obs = self.observation_model.create_observation(player, self.current_game.players)
            observations[agent_id] = np.array(obs, dtype=np.float32)
            infos[agent_id] = {
                "accuracy": player.accuracy,
                "alive": player.alive
            }
        
        # Update previous alive state
        self.prev_alive_state = {
            player.id: player.alive for player in self.current_game.players
        }
        
        return observations, infos

    def step(self, action_dict: Dict[int, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one environment step with the given actions."""
        shots = self.current_game.execute_turn(action_dict)
        self.current_game.prepare_turn()
        
        # Get observations, rewards, and done flags
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}
        
        game_over = self.current_game.is_over()
        rewards = self._calculate_rewards(shots, game_over)
        
        for player in self.current_game.players:
            agent_id = player.id
            
            # Observation
            obs = self.observation_model.create_observation(player, self.current_game.players)
            observations[agent_id] = np.array(obs, dtype=np.float32)
            
            # Reward calculation
            rewards[agent_id] = rewards.get(agent_id, 0.0)
            
            # Termination and truncation
            terminateds[agent_id] = not player.alive
            truncateds[agent_id] = False
            
            # Info
            infos[agent_id] = {
                "accuracy": player.accuracy,
                "alive": player.alive
            }
        
        # Global termination condition
        terminateds["__all__"] = game_over
        truncateds["__all__"] = False
        
        # Update previous alive state for next step
        self.prev_alive_state = {
            player.id: player.alive for player in self.current_game.players
        }
        
        return observations, rewards, terminateds, truncateds, infos

    def get_observation_space(self, agent_id: str) -> gym.Space:
        """Get observation space for a specific agent."""
        return self.observation_spaces[agent_id]

    def get_action_space(self, agent_id: str) -> gym.Space:
        """Get action space for a specific agent."""
        return self.action_spaces[agent_id]

    def render(self) -> None:
        """Render the environment."""
        run_infinite_game_visual(self)

    def close(self) -> None:
        """Clean up resources."""
        self.current_game = None
        self.prev_alive_state = {}