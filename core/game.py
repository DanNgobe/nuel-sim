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
        self.already_shot: set[int] = set()
        self.history = []  # [[(shooter, target, hit), ...], ...]
       
        # Round Management
        self.max_rounds = max_rounds
        self.round_number = 0

    @property
    def rounds_remaining(self):
        if self.max_rounds is None:
            return None
        return self.max_rounds - self.round_number

    def get_alive_players(self):
        return [p for p in self.players if p.alive]
    
    def is_over(self):
        game_over = self.gameplay.is_over(self.players)
        rounds_exceeded = self.rounds_remaining is not None and self.rounds_remaining <= 0
        if game_over or rounds_exceeded:
            self.observation_model.reset()
        return game_over or rounds_exceeded


    def run_turn(self):
        # Prepare eligible shooters
        alive_players = self.get_alive_players()
        eligible_players = [p for p in alive_players if p.id not in self.already_shot]
        if not eligible_players:
            eligible_players = alive_players
            self.already_shot = set()
            self.round_number += 1

        # Phase 1: Choose shooters and conduct shots
        shooters = self.gameplay.choose_shooters(eligible_players)
        shots = self.gameplay.conduct_shots(shooters, self.players)
        self.gameplay.process_shots(shots)

        # Phase 2: Update observation model and history
        self.history.append(shots)
        for shooter in shooters:
            self.already_shot.add(shooter.id)
        self.observation_model.update(shots, self.rounds_remaining)
       