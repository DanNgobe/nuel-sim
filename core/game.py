from core.player_manager import PlayerManager
from core.observation import NullObservationModel
from core.gameplay import GamePlay
from core.player import Player
from typing import List, Optional, Tuple, Dict, Any

class Game:
    def __init__(
        self, 
        players: List[Player], 
        gameplay: GamePlay, 
        observation_model: Optional[Any] = NullObservationModel(), 
        max_rounds: Optional[int] = None
    ):
        self.players = players
        self.gameplay = gameplay
        self.observation_model = observation_model

        # Initialize the observation model
        self.observation_model.initialize(players)
        
        # History Management
        self.history = []  # [[(shooter, target, hit), ...], ...]
       
        # Round Management
        self.max_rounds = max_rounds
        self.round_number = 0

        # Player Management
        self.player_manager = PlayerManager(players)
        
        # Current turn state
        self.current_shooters = []

    @property
    def rounds_remaining(self) -> Optional[int]:
        """Get the number of rounds remaining"""
        if self.max_rounds is None:
            return None
        return self.max_rounds - self.round_number

    def get_alive_players(self) -> List[Player]:
        """Get all alive players"""
        return self.player_manager.get_alive_players()
    
    def is_over(self) -> bool:
        """Check if the game is over"""
        game_over = self.gameplay.is_over(self.players, self.get_alive_players())
        rounds_exceeded = self.rounds_remaining is not None and self.rounds_remaining <= 0
                 
        return game_over or rounds_exceeded

    def prepare_turn(self) -> List[Player]:
        """Prepare for a new turn and return eligible players"""
        eligible_players = self.player_manager.get_eligible_players()
        if not eligible_players:
            self.round_number += 1
            eligible_players = self.player_manager.get_alive_players()
            self.player_manager.reset_already_shot()
        
        # Get eligible players for this turn
        shooters = self.gameplay.choose_shooters(eligible_players)
        self.current_shooters = shooters
        return self.current_shooters

    def execute_turn(self, actions: Dict[Player, Player]) -> List[Tuple[Player, Optional[Player], bool]]:
        """
        Execute a turn with provided actions
        
        Args:
            actions: Dictionary mapping player to target player
            
        Returns:
            List of shot results (shooter, target, hit)
        """
        shooters = actions.keys()
        shots = []
        
        # Process each action
        for shooter, target in actions.items():
            # Validate and execute shot
            if shooter and shooter.alive and shooter in self.current_shooters:
                hit = shooter.shoot(target)
                shots.append((shooter, target, hit))

        
        # Process the shots
        self.gameplay.process_shots(shots)
        
        # Update game state
        self.history.append(shots)
        self.player_manager.mark_shot(shooters)
        self.observation_model.update(shots, self.rounds_remaining)
        
        return shots

    def run_auto_turn(self) -> List[Tuple[Player, Optional[Player], bool]]:
        """Run a turn using players' default strategies"""
        eligible_players = self.prepare_turn()
        
        # Get actions from player strategies
        actions = {}
        for player in eligible_players:
            target = player.choose_target(self.players)
            actions[player] = target
        
        # Execute the turn
        return self.execute_turn(actions)