class PlayerManager:
    def __init__(self, players):
        self.players = players
        self.already_shot: set[int] = set()

    def get_alive_players(self):
        return [p for p in self.players if p.alive]

    def get_eligible_players(self):
        alive_players = self.get_alive_players()
        eligible_players = [p for p in alive_players if p.id not in self.already_shot]
        return eligible_players

    def mark_shot(self, shooters):
        for shooter in shooters:
            self.already_shot.add(shooter.id)

    def reset_already_shot(self):
        self.already_shot = set()