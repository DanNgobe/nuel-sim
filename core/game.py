from core.player_manager import PlayerManager
from .observation import NullObservationModel
from .gameplay import GamePlay
from .player import Player

class Game:
    def __init__(self, players: list[Player], gameplay: GamePlay, observation_model=NullObservationModel(), max_rounds=None):
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

    @property
    def rounds_remaining(self):
        if self.max_rounds is None:
            return None
        return self.max_rounds - self.round_number

    def get_alive_players(self):
        return self.player_manager.get_alive_players()
    
    def is_over(self):
        game_over = self.gameplay.is_over(self.players, self.get_alive_players())
        rounds_exceeded = self.rounds_remaining is not None and self.rounds_remaining <= 0
        if game_over or rounds_exceeded:
            self.observation_model.reset()
        return game_over or rounds_exceeded

    def run_turn(self):
        # Prepare eligible shooters
        eligible_players = self.player_manager.get_eligible_players()
        if not eligible_players:
            self.round_number += 1
            eligible_players = self.player_manager.get_alive_players()
            self.player_manager.reset_already_shot()
        
        # Phase 1: Choose shooters and conduct shots
        shooters = self.gameplay.choose_shooters(eligible_players)
        shots = self.gameplay.conduct_shots(shooters, self.players)
        self.gameplay.process_shots(shots)

        # Phase 2: Update observation model and history
        self.history.append(shots)
        self.player_manager.mark_shot(shooters)
        self.observation_model.update(shots, self.rounds_remaining)