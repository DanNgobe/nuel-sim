import random


def target_weakest(me, players):
    alive = [p for p in players if p != me and p.alive]
    return min(alive, key=lambda p: p.accuracy, default=None)

def target_strongest(me, players):
    alive = [p for p in players if p != me and p.alive]
    return max(alive, key=lambda p: p.accuracy, default=None)

def target_stronger(me, players):
    alive = [p for p in players if p != me and p.alive]
    return max(alive, key=lambda p: p.accuracy - me.accuracy, default=None)

def target_random(me, players):
    alive = [p for p in players if p != me and p.alive]
    return random.choice(alive) if alive else None


# Xu's strategy (truels)
def target_stronger_or_strongest(me, players): 
    stronger = target_stronger(me, players)
    strongest = target_strongest(me, players)
    
    # if the second strongest is decently strong target him so if he kills the strongest we can kill him
    if stronger and stronger.accuracy > 0.65:
        return stronger
    return strongest


def target_nearest(me, players):
    alive = [p for p in players if p != me and p.alive]
    return min(alive, key=lambda p: (p.x - me.x)**2 + (p.y - me.y)**2, default=None)
