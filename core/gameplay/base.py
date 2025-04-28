class GamePlay:

    def is_over(self, players):
        alive_players = [player for player in players if player.alive]
        return len(alive_players) <= 1

    def choose_shooters(self, alive_players, last_shooter):
        raise NotImplementedError

    def conduct_shots(self, shooters, players):
        shots = []
        for shooter in shooters:
            target = shooter.choose_target(players)
            hit = shooter.shoot(target)
            shots.append((shooter, target, hit))
        return shots

    def process_shots(self, shots):
        for shooter, target, hit in shots:
            if hit and target and target.alive:
                target.alive = False
