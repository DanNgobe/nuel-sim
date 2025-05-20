from .base import GamePlay

# Players take turns shooting in a sequence
class SequentialGamePlay(GamePlay):
    def choose_shooters(self, eligible_players):
        shooter = eligible_players[0]
        return [shooter]