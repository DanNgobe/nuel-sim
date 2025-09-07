import pygame
import time
import math
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, PLAYER_RADIUS, FPS, ARROW_SIZE,
    COLOR_BACKGROUND, COLOR_PLAYER_ALIVE, COLOR_PLAYER_DEAD,
    COLOR_HIT, COLOR_MISS
)

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
    screen.fill(COLOR_BACKGROUND)

    # Draw players
    for p in players:
        color = COLOR_PLAYER_ALIVE if p.alive else COLOR_PLAYER_DEAD
        pygame.draw.circle(screen, color, (int(p.x), int(p.y)), PLAYER_RADIUS)

        font = pygame.font.SysFont(None, 24)

        # Render player name above the circle
        name_label = font.render(p.name, True, (255, 255, 255))
        screen.blit(name_label, (p.x - PLAYER_RADIUS, p.y - PLAYER_RADIUS - 20))

        # Render marksmanship inside the circle (centered)
        accuracy_text = f"{int(p.accuracy * 100)}%"
        acc_label = font.render(accuracy_text, True, (255, 255, 255))
        text_rect = acc_label.get_rect(center=(p.x, p.y))
        screen.blit(acc_label, text_rect)


    # Draw last rounds shots (shooter -> target)
    if history:
        for shooter, target, hit in history[-1]:
            # Draw arrow between shooter and target
            if shooter and target:
                # Draw arrow between shooter and target
                line_color = COLOR_HIT if hit else COLOR_MISS
                draw_arrow(screen, (shooter.x, shooter.y), (target.x, target.y), line_color)
    pygame.display.flip()

def run_game_visual(game):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
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

def run_infinite_game_visual(game_manager):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Nuel Simulation")
    clock = pygame.time.Clock()

    while True:
        game = game_manager.reset_game()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
        while not game.is_over():
            game.run_auto_turn()
            clock.tick(FPS)
            draw_game(screen, game.players, game.history)

        alive = game.get_alive_players()
        if alive:
            print(f"Winner: {alive[0].name}")
        else:
            print("No survivors.")
        time.sleep(1)
        