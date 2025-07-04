import pygame
import math
import random

# --- Constants ---
WIDTH, HEIGHT = 800, 800
BLACK_HOLE_MASS = 5e5  # Arbitrary mass unit
G = 6.67430e-1         # Scaled gravitational constant for visual effect
NUM_STARS = 2
EVENT_HORIZON_RADIUS = 100
STAR_MASS = 1e2  # Arbitrary mass unit for stars

# --- Initialize Pygame ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Black Hole Spacetime Curvature")
clock = pygame.time.Clock()

# --- Black Hole Position ---
BH_X = WIDTH // 2
BH_Y = HEIGHT // 2

# --- Star Class ---
class Star:
    def __init__(self):
        self.mass = STAR_MASS
        self.reset()

    def reset(self):
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(100, 400)
        self.x = BH_X + distance * math.cos(angle)
        self.y = BH_Y + distance * math.sin(angle)
        speed = math.sqrt(G * BLACK_HOLE_MASS / distance)
        # Tangential velocity for approximate orbit
        self.vx = -speed * math.sin(angle)
        self.vy = speed * math.cos(angle)
        self.color = (255, 255, 255)

    def update(self):
        # Vector from star to black hole
        dx = BH_X - self.x
        dy = BH_Y - self.y
        distance = math.hypot(dx, dy)
        force = G * BLACK_HOLE_MASS / (distance ** 2 + 1e-3)
        # Normalize direction
        fx = force * dx / distance
        fy = force * dy / distance

        # Update velocity and position
        self.vx += fx
        self.vy += fy
        self.x += self.vx
        self.y += self.vy

        # If too close, respawn and signal energy release
        if distance < EVENT_HORIZON_RADIUS:
            energy = G * BLACK_HOLE_MASS * self.mass / EVENT_HORIZON_RADIUS
            flashes.append({'energy': energy, 'timer': 30})  # flash lasts 30 frames
            self.reset()

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), 2)

# --- Create stars ---
stars = [Star() for _ in range(NUM_STARS)]
flashes = []

# --- Main loop ---
running = True
while running:
    clock.tick(60)
    screen.fill((0, 0, 0))

    # Draw black hole event horizon
    pygame.draw.circle(screen, (0, 0, 0), (BH_X, BH_Y), EVENT_HORIZON_RADIUS)
    pygame.draw.circle(screen, (50, 50, 50), (BH_X, BH_Y), EVENT_HORIZON_RADIUS, 2)
    pygame.draw.circle(screen, (100, 100, 100), (BH_X, BH_Y), 60, 1)

    # Draw energy flashes
    for flash in flashes[:]:
        alpha = int(255 * (flash['timer'] / 30))
        flash_color = (255, 255, 100, alpha)
        flash_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(flash_surface, flash_color, (BH_X, BH_Y), EVENT_HORIZON_RADIUS + 10)
        screen.blit(flash_surface, (0, 0))
        # Show energy value
        font = pygame.font.SysFont(None, 28)
        energy_text = font.render(f"Energy Released: {flash['energy']:.1e}", True, (255, 255, 0))
        screen.blit(energy_text, (BH_X - 120, BH_Y - EVENT_HORIZON_RADIUS - 40))
        flash['timer'] -= 1
        if flash['timer'] <= 0:
            flashes.remove(flash)

    for star in stars:
        star.update()
        star.draw(screen)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()

pygame.quit()
