from .base import GamePlay

# All players shoot at the same time
class SimultaneousGamePlay(GamePlay):
    def choose_shooters(self, eligible_players):
        return eligible_players[:]  # all eligible players shoot at the same time

    def process_shots(self, shots):
        # Phase 2: resolve shots after all shooting
        dead_players = set()
        for shooter, target, hit in shots:
            if hit and target and target.alive:
                dead_players.add(target)
        
        for player in dead_players:
            player.alive = False
