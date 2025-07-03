import pygame
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bouncing Ball (Enhanced)")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Ball parameters
ball_radius = 30
ball_x = WIDTH // 2
ball_y = 50
ball_velocity_y = 0

# Physics parameters
gravity = 0.5
bounce_factor = 0.7

# Font for displaying info
font = pygame.font.SysFont(None, 32)

def draw_info():
    gravity_text = font.render(f"Gravity: {gravity:.2f}", True, BLACK)
    bounce_text = font.render(f"Bounce: {bounce_factor:.2f}", True, BLACK)
    radius_text = font.render(f"Radius: {ball_radius}", True, BLACK)
    screen.blit(gravity_text, (10, 10))
    screen.blit(bounce_text, (10, 40))
    screen.blit(radius_text, (10, 70))

# Main game loop
clock = pygame.time.Clock()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        # User controls
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Reset ball
                ball_y = 50
                ball_velocity_y = 0
            if event.key == pygame.K_UP:
                gravity = min(gravity + 0.1, 5)
            if event.key == pygame.K_DOWN:
                gravity = max(gravity - 0.1, 0.1)
            if event.key == pygame.K_RIGHT:
                bounce_factor = min(bounce_factor + 0.05, 1)
            if event.key == pygame.K_LEFT:
                bounce_factor = max(bounce_factor - 0.05, 0)
            if event.key == pygame.K_s:
                ball_radius = min(ball_radius + 5, 200)
            if event.key == pygame.K_a:
                ball_radius = max(ball_radius - 5, 5)

    # Update physics
    ball_velocity_y += gravity
    ball_y += ball_velocity_y

    # Check for bounce with energy loss
    if ball_y > HEIGHT - ball_radius:
        ball_y = HEIGHT - ball_radius
        ball_velocity_y = -ball_velocity_y * bounce_factor

    # Prevent tiny bounces
    if abs(ball_velocity_y) < 0.5 and ball_y >= HEIGHT - ball_radius:
        ball_velocity_y = 0
        ball_y = HEIGHT - ball_radius

    # Draw everything
    screen.fill(WHITE)
    pygame.draw.circle(screen, RED, (int(ball_x), int(ball_y)), ball_radius)
    draw_info()

    pygame.display.flip()
    clock.tick(60)