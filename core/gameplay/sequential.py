from .base import GamePlay

# Players take turns shooting in a sequence
class SequentialGamePlay(GamePlay):
    def choose_shooters(self, alive_players, last_shooter):
        if not alive_players:
            return []
        if last_shooter not in alive_players:
            return [alive_players[0]]
        idx = (alive_players.index(last_shooter) + 1) % len(alive_players)
        return [alive_players[idx]]
