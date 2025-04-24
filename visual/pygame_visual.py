import pygame
import time
import math

WIDTH, HEIGHT = 800, 600
RADIUS = 30
SHOT_COLOR_HIT = (0, 255, 0)
SHOT_COLOR_MISS = (255, 0, 0)
BACKGROUND = (30, 30, 30)
FPS = 1

ARROW_SIZE = 15  # Size of the arrowhead

def draw_arrow(screen, start_pos, end_pos, color):
    """Draws an arrow from start_pos to end_pos."""
    pygame.draw.line(screen, color, start_pos, end_pos, 4)

    # Calculate the angle of the arrow
    angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])

    # Arrowhead points (triangle)
    arrowhead = [
        (end_pos[0] - ARROW_SIZE * math.cos(angle - math.pi / 6), end_pos[1] - ARROW_SIZE * math.sin(angle - math.pi / 6)),
        (end_pos[0] - ARROW_SIZE * math.cos(angle + math.pi / 6), end_pos[1] - ARROW_SIZE * math.sin(angle + math.pi / 6)),
        end_pos
    ]
    
    # Draw the arrowhead
    pygame.draw.polygon(screen, color, arrowhead)

def draw_game(screen, players, history):
    screen.fill(BACKGROUND)

    # Draw players
    for p in players:
        color = (0, 150, 255) if p.alive else (100, 100, 100)
        pygame.draw.circle(screen, color, (int(p.x), int(p.y)), RADIUS)
        font = pygame.font.SysFont(None, 24)
        label = font.render(p.name, True, (255, 255, 255))
        screen.blit(label, (p.x - RADIUS, p.y - RADIUS - 20))

    # Draw last shot (shooter -> target)
    if history:
        shooter, target, hit = history[-1]
        if shooter and target:
            # Draw arrow between shooter and target
            line_color = SHOT_COLOR_HIT if hit else SHOT_COLOR_MISS
            draw_arrow(screen, (shooter.x, shooter.y), (target.x, target.y), line_color)
    pygame.display.flip()

def run_game_visual(game):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Nuel Simulation")
    clock = pygame.time.Clock()

    time.sleep(1)  # Pause for a moment before starting

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game.is_over():
            game.run_turn()
        else:
            alive = game.get_alive_players()
            if alive:
                print(f"Winner: {alive[0].name}")
            else:
                print("No survivors.")
            time.sleep(2)
            running = False

        draw_game(screen, game.players, game.history)

    pygame.quit()
