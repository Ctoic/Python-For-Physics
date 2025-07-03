"""
2D Elastic Collision Simulator
A comprehensive physics simulation demonstrating momentum and energy conservation
with object-oriented design and performance optimization.
"""

import pygame
import numpy as np
import math
import random
from typing import List, Tuple
import time
from dataclasses import dataclass


@dataclass
class PhysicsStats:
    """Track physics quantities for verification"""
    total_momentum: Tuple[float, float]
    total_kinetic_energy: float
    
    
class Ball:
    """Represents a ball with physics properties"""
    
    def __init__(self, x: float, y: float, vx: float, vy: float, 
                 mass: float, radius: float, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
        self.radius = radius
        self.color = color
        self.trail = []  # For visual trail effect
        self.max_trail_length = 20
        
    def update(self, dt: float):
        """Update position based on velocity"""
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Add to trail
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)
    
    def get_momentum(self) -> Tuple[float, float]:
        """Calculate momentum vector"""
        return (self.mass * self.vx, self.mass * self.vy)
    
    def get_kinetic_energy(self) -> float:
        """Calculate kinetic energy"""
        return 0.5 * self.mass * (self.vx**2 + self.vy**2)
    
    def distance_to(self, other: 'Ball') -> float:
        """Calculate distance to another ball"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def is_colliding_with(self, other: 'Ball') -> bool:
        """Check if colliding with another ball"""
        return self.distance_to(other) <= (self.radius + other.radius)


class CollisionSimulator:
    """Main simulation class with optimized collision detection"""
    
    def __init__(self, width: int = 1000, height: int = 600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("2D Elastic Collision Simulator")
        self.clock = pygame.time.Clock()
        
        # Physics settings
        self.gravity = 0  # Can be enabled for more complex scenarios
        self.damping = 1.0  # Energy loss factor (1.0 = no loss)
        self.wall_bounce = True
        
        # Visual settings
        self.show_vectors = True
        self.show_trails = True
        self.show_stats = True
        
        # Performance tracking
        self.collision_checks = 0
        self.frame_times = []
        
        # Initialize balls
        self.balls = []
        self.setup_default_scenario()
        
    def setup_default_scenario(self):
        """Setup a default interesting collision scenario"""
        self.balls = [
            Ball(200, 300, 100, 0, 5, 25, (255, 100, 100)),  # Red ball
            Ball(800, 300, -80, 0, 3, 20, (100, 255, 100)),  # Green ball
            Ball(500, 150, 0, 60, 4, 22, (100, 100, 255)),   # Blue ball
            Ball(500, 450, 0, -60, 6, 28, (255, 255, 100)),  # Yellow ball
        ]
    
    def add_random_ball(self):
        """Add a random ball to the simulation"""
        x = random.randint(50, self.width - 50)
        y = random.randint(50, self.height - 50)
        vx = random.randint(-100, 100)
        vy = random.randint(-100, 100)
        mass = random.uniform(2, 8)
        radius = random.randint(15, 30)
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        
        new_ball = Ball(x, y, vx, vy, mass, radius, color)
        
        # Ensure it doesn't overlap with existing balls
        for ball in self.balls:
            if new_ball.is_colliding_with(ball):
                return  # Don't add if it would overlap
        
        self.balls.append(new_ball)
    
    def handle_wall_collisions(self, ball: Ball):
        """Handle collisions with walls"""
        if ball.x - ball.radius <= 0 or ball.x + ball.radius >= self.width:
            ball.vx *= -self.damping
            ball.x = max(ball.radius, min(self.width - ball.radius, ball.x))
        
        if ball.y - ball.radius <= 0 or ball.y + ball.radius >= self.height:
            ball.vy *= -self.damping
            ball.y = max(ball.radius, min(self.height - ball.radius, ball.y))
    
    def handle_ball_collision(self, ball1: Ball, ball2: Ball):
        """Handle elastic collision between two balls"""
        # Calculate collision normal
        dx = ball2.x - ball1.x
        dy = ball2.y - ball1.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance == 0:  # Avoid division by zero
            return
        
        # Normalize collision vector
        nx = dx / distance
        ny = dy / distance
        
        # Separate balls if overlapping
        overlap = (ball1.radius + ball2.radius) - distance
        if overlap > 0:
            ball1.x -= nx * overlap * 0.5
            ball1.y -= ny * overlap * 0.5
            ball2.x += nx * overlap * 0.5
            ball2.y += ny * overlap * 0.5
        
        # Calculate relative velocity
        dvx = ball2.vx - ball1.vx
        dvy = ball2.vy - ball1.vy
        
        # Calculate relative velocity along collision normal
        dvn = dvx * nx + dvy * ny
        
        # Don't resolve if objects are separating
        if dvn > 0:
            return
        
        # Calculate collision impulse
        impulse = 2 * dvn / (ball1.mass + ball2.mass)
        
        # Update velocities
        ball1.vx += impulse * ball2.mass * nx
        ball1.vy += impulse * ball2.mass * ny
        ball2.vx -= impulse * ball1.mass * nx
        ball2.vy -= impulse * ball1.mass * ny
    
    def detect_collisions_naive(self):
        """Naive O(n²) collision detection"""
        self.collision_checks = 0
        for i in range(len(self.balls)):
            for j in range(i + 1, len(self.balls)):
                self.collision_checks += 1
                if self.balls[i].is_colliding_with(self.balls[j]):
                    self.handle_ball_collision(self.balls[i], self.balls[j])
    
    def detect_collisions_optimized(self):
        """Optimized collision detection using spatial partitioning"""
        self.collision_checks = 0
        
        # Simple spatial hash grid
        grid_size = 100
        grid = {}
        
        # Place balls in grid
        for ball in self.balls:
            gx = int(ball.x // grid_size)
            gy = int(ball.y // grid_size)
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (gx + dx, gy + dy)
                    if key not in grid:
                        grid[key] = []
                    if ball not in grid[key]:
                        grid[key].append(ball)
        
        # Check collisions within each grid cell
        checked_pairs = set()
        for cell_balls in grid.values():
            for i in range(len(cell_balls)):
                for j in range(i + 1, len(cell_balls)):
                    ball1, ball2 = cell_balls[i], cell_balls[j]
                    pair = (id(ball1), id(ball2))
                    if pair not in checked_pairs:
                        checked_pairs.add(pair)
                        self.collision_checks += 1
                        if ball1.is_colliding_with(ball2):
                            self.handle_ball_collision(ball1, ball2)
    
    def calculate_physics_stats(self) -> PhysicsStats:
        """Calculate total momentum and kinetic energy"""
        total_px = sum(ball.get_momentum()[0] for ball in self.balls)
        total_py = sum(ball.get_momentum()[1] for ball in self.balls)
        total_ke = sum(ball.get_kinetic_energy() for ball in self.balls)
        
        return PhysicsStats((total_px, total_py), total_ke)
    
    def draw_ball(self, ball: Ball):
        """Draw a ball with trail and vectors"""
        # Draw trail
        if self.show_trails and len(ball.trail) > 1:
            for i in range(len(ball.trail) - 1):
                alpha = i / len(ball.trail)
                trail_color = tuple(int(c * alpha) for c in ball.color)
                pygame.draw.circle(self.screen, trail_color, 
                                 (int(ball.trail[i][0]), int(ball.trail[i][1])), 
                                 max(1, int(ball.radius * alpha * 0.3)))
        
        # Draw ball
        pygame.draw.circle(self.screen, ball.color, 
                         (int(ball.x), int(ball.y)), int(ball.radius))
        
        # Draw velocity vector
        if self.show_vectors:
            vector_scale = 0.5
            end_x = ball.x + ball.vx * vector_scale
            end_y = ball.y + ball.vy * vector_scale
            pygame.draw.line(self.screen, (255, 255, 255), 
                           (ball.x, ball.y), (end_x, end_y), 2)
            
            # Draw arrowhead
            angle = math.atan2(ball.vy, ball.vx)
            arrow_length = 10
            arrow_angle = math.pi / 6
            
            arrow_x1 = end_x - arrow_length * math.cos(angle - arrow_angle)
            arrow_y1 = end_y - arrow_length * math.sin(angle - arrow_angle)
            arrow_x2 = end_x - arrow_length * math.cos(angle + arrow_angle)
            arrow_y2 = end_y - arrow_length * math.sin(angle + arrow_angle)
            
            pygame.draw.line(self.screen, (255, 255, 255), 
                           (end_x, end_y), (arrow_x1, arrow_y1), 2)
            pygame.draw.line(self.screen, (255, 255, 255), 
                           (end_x, end_y), (arrow_x2, arrow_y2), 2)
    
    def draw_ui(self, stats: PhysicsStats, fps: float):
        """Draw UI elements and statistics"""
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 18)
        
        # Background for stats
        stats_bg = pygame.Surface((300, 200))
        stats_bg.set_alpha(128)
        stats_bg.fill((0, 0, 0))
        self.screen.blit(stats_bg, (10, 10))
        
        # Physics stats
        y_offset = 20
        texts = [
            f"Balls: {len(self.balls)}",
            f"FPS: {fps:.1f}",
            f"Collision Checks: {self.collision_checks}",
            f"Momentum X: {stats.total_momentum[0]:.2f}",
            f"Momentum Y: {stats.total_momentum[1]:.2f}",
            f"Total KE: {stats.total_kinetic_energy:.2f}",
        ]
        
        for text in texts:
            surface = small_font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (20, y_offset))
            y_offset += 20
        
        # Instructions
        instructions = [
            "SPACE: Add random ball",
            "R: Reset simulation",
            "V: Toggle vectors",
            "T: Toggle trails",
            "S: Toggle stats",
            "ESC: Quit"
        ]
        
        y_offset = self.height - 140
        for instruction in instructions:
            surface = small_font.render(instruction, True, (255, 255, 255))
            self.screen.blit(surface, (20, y_offset))
            y_offset += 20
    
    def run(self, use_optimized_collision=True):
        """Main simulation loop"""
        running = True
        dt = 0.016  # 60 FPS target
        
        while running:
            frame_start = time.time()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.add_random_ball()
                    elif event.key == pygame.K_r:
                        self.setup_default_scenario()
                    elif event.key == pygame.K_v:
                        self.show_vectors = not self.show_vectors
                    elif event.key == pygame.K_t:
                        self.show_trails = not self.show_trails
                    elif event.key == pygame.K_s:
                        self.show_stats = not self.show_stats
            
            # Update physics
            for ball in self.balls:
                ball.update(dt)
                if self.wall_bounce:
                    self.handle_wall_collisions(ball)
            
            # Handle collisions
            if use_optimized_collision:
                self.detect_collisions_optimized()
            else:
                self.detect_collisions_naive()
            
            # Calculate physics stats
            stats = self.calculate_physics_stats()
            
            # Render
            self.screen.fill((20, 20, 40))
            
            for ball in self.balls:
                self.draw_ball(ball)
            
            if self.show_stats:
                fps = self.clock.get_fps()
                self.draw_ui(stats, fps)
            
            pygame.display.flip()
            self.clock.tick(60)
            
            # Track frame time
            frame_time = time.time() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 60:
                self.frame_times.pop(0)
        
        pygame.quit()


def performance_comparison():
    """Compare performance of naive vs optimized collision detection"""
    print("Performance Comparison: Naive vs Optimized Collision Detection")
    print("=" * 60)
    
    # Test with different numbers of balls
    test_sizes = [10, 20, 50, 100]
    
    for size in test_sizes:
        print(f"\nTesting with {size} balls:")
        
        # Create simulator
        sim = CollisionSimulator()
        
        # Add balls
        for _ in range(size):
            sim.add_random_ball()
        
        # Test naive approach
        start_time = time.time()
        iterations = 100
        for _ in range(iterations):
            sim.detect_collisions_naive()
        naive_time = (time.time() - start_time) / iterations
        naive_checks = sim.collision_checks
        
        # Test optimized approach
        start_time = time.time()
        for _ in range(iterations):
            sim.detect_collisions_optimized()
        optimized_time = (time.time() - start_time) / iterations
        optimized_checks = sim.collision_checks
        
        print(f"  Naive:     {naive_time*1000:.3f}ms, {naive_checks} checks")
        print(f"  Optimized: {optimized_time*1000:.3f}ms, {optimized_checks} checks")
        print(f"  Speedup:   {naive_time/optimized_time:.2f}x")


if __name__ == "__main__":
    print("2D Elastic Collision Simulator")
    print("Choose an option:")
    print("1. Run simulation")
    print("2. Performance comparison")
    print("3. Both")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "2":
        performance_comparison()
    elif choice == "3":
        performance_comparison()
        print("\nStarting simulation...")
        sim = CollisionSimulator()
        sim.run(use_optimized_collision=True)
    else:
        sim = CollisionSimulator()
        sim.run(use_optimized_collision=True)