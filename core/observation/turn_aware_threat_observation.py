from .threat_level_observation import ThreatLevelObservation

class TurnAwareThreatObservation(ThreatLevelObservation):
    def __init__(self, num_players, has_ghost=False):
        super().__init__(num_players, has_ghost)

    @property
    def name(self) -> str:
        return "TurnAwareThreatObservation" + ("_with_abstention" if self.has_ghost else "")

    def create_observation(self, player, players):
        obs = [player.accuracy]

        alive_non_ghost = [p for p in players if p.alive and p.name != "Ghost"]
        if player not in alive_non_ghost:
            return obs + [0.0, 0.0] * (self.num_players - 1)

        player_index = alive_non_ghost.index(player)

        others = []
        for p in players:
            if p == player or p.name == "Ghost":
                continue

            acc = p.accuracy if p.alive else 0.0

            if p.alive and p in alive_non_ghost:
                i = alive_non_ghost.index(p)
                turn_dist = (i - player_index) % len(alive_non_ghost)
                turn_dist = max(turn_dist, 1)
            else:
                turn_dist = 0.0

            others.append((turn_dist, acc, p))

        # Sort by accuracy descending
        others.sort(key=lambda x: x[1], reverse=True)

        for turn_dist, acc, _ in others:
            obs.extend([turn_dist, acc])
        return obs

    def get_observation_dim(self):
        return 1 + 2 * (self.num_players - 1)

    def get_targets(self, player, players):
        # Return targets sorted by accuracy descending (to match observation)
        others = [p for p in players if p != player]
        others.sort(key=lambda p: p.accuracy if p.alive else 0.0, reverse=True)
        return others
