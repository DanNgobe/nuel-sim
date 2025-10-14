from .simple_observation import SimpleObservation

class SortedObservation(SimpleObservation):
    def __init__(self, num_players, has_ghost=False):
        super().__init__(num_players, has_ghost)

    @property
    def name(self) -> str:
        return "SortedObservation" + ("_with_abstention" if self.has_ghost else "")

    def create_observation(self, player, players):
        obs = [player.accuracy]  # Self accuracy

        others = [p for p in players if (p != player and p.name != "Ghost")]
        # Sort by threat level: alive * accuracy
        others.sort(key=lambda p: (p.accuracy if p.alive else 0.0), reverse=True)

        for p in others:
            obs.append(p.accuracy if p.alive else 0.0)
        return obs