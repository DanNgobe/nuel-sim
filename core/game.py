
class Game:
    def __init__(self, players, gameplay):
        self.players = players
        self.history = [] # [[(shooter, target, hit), ...], ...]
        self.last_shooter = None
        self.gameplay = gameplay

    def get_alive_players(self):
        return [p for p in self.players if p.alive]

    def is_over(self):
        return self.gameplay.is_over(self.players)

    def run_turn(self):
        alive = self.get_alive_players()
        shooters = self.gameplay.choose_shooters(alive, self.last_shooter)
        shots = self.gameplay.conduct_shots(shooters, self.players)
        self.gameplay.process_shots(shots)
        self.history.append(shots)
        if shooters:
            self.last_shooter = shooters[-1]
