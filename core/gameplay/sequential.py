from .base import GamePlay

# Players take turns shooting in a sequence
class SequentialGamePlay(GamePlay):
    def __init__(self):
        super().__init__()
        self.already_shot = set()

    def choose_shooters(self, alive_players, last_shooter=None):
        eligible_players = [p for p in alive_players if p not in self.already_shot]
        if not eligible_players:
            self.already_shot.clear()
            eligible_players = alive_players

        shooter = eligible_players[0]
        self.already_shot.add(shooter)
        return [shooter]
    
    def is_over(self, players):
        over = super().is_over(players)
        if over:
            self.already_shot.clear()
        return over

