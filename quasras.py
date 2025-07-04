import pygame
import numpy as np
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60
G = 6.67430e-11  # Gravitational constant (scaled for simulation)
SIMULATION_G = 1000  # Scaled gravitational constant for visual effect

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 100, 255)
RED = (255, 50, 50)
YELLOW = (255, 255, 100)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
GREEN = (0, 255, 0)

@dataclass
class Particle:
    """Represents a particle in the accretion disk"""
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    radius: float
    color: Tuple[int, int, int]
    lifetime: float
    max_lifetime: float

class Quasar:
    def __init__(self, x, y, mass, color, name):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.mass = mass
        self.color = color
        self.name = name
        self.radius = max(10, math.sqrt(mass) * 0.5)
        self.accretion_particles = []
        self.jet_particles = []
        self.trail_points = []
        self.max_trail_length = 100
        
        # Initialize accretion disk particles
        self.init_accretion_disk()
        
    def init_accretion_disk(self):
        """Initialize particles in the accretion disk"""
        num_particles = 50
        for _ in range(num_particles):
            # Random distance from center (accretion disk radius)
            disk_radius = random.uniform(self.radius * 2, self.radius * 8)
            angle = random.uniform(0, 2 * math.pi)
            
            # Position relative to quasar
            px = math.cos(angle) * disk_radius
            py = math.sin(angle) * disk_radius
            
            # Orbital velocity (simplified Keplerian motion)
            orbital_speed = math.sqrt(SIMULATION_G * self.mass / disk_radius) * 0.1
            vx = -math.sin(angle) * orbital_speed
            vy = math.cos(angle) * orbital_speed
            
            # Color based on distance (hotter = closer)
            heat_factor = max(0.0, 1.0 - (disk_radius - self.radius * 2) / (self.radius * 6))
            if heat_factor > 0.7:
                color = (255, 255, max(0, min(255, int(255 * heat_factor))))
            elif heat_factor > 0.4:
                color = (255, max(0, min(255, int(255 * heat_factor))), 0)
            else:
                color = (max(0, min(255, int(255 * heat_factor))), 0, 0)
            
            particle = Particle(
                x=px, y=py, vx=vx, vy=vy,
                angle=angle, radius=disk_radius,
                color=color, lifetime=0, max_lifetime=random.uniform(300, 600)
            )
            self.accretion_particles.append(particle)
    
    def update_accretion_disk(self, other_quasar=None):
        """Update accretion disk particles"""
        for particle in self.accretion_particles[:]:
            # Update lifetime
            particle.lifetime += 1
            
            # Remove old particles
            if particle.lifetime > particle.max_lifetime:
                self.accretion_particles.remove(particle)
                continue
            
            # Distance from quasar center
            dx = particle.x
            dy = particle.y
            distance = math.sqrt(dx**2 + dy**2)
            
            # Gravitational acceleration toward quasar center
            if distance > 0:
                acc_magnitude = SIMULATION_G * self.mass / (distance**2)
                acc_x = -(dx / distance) * acc_magnitude * 0.01
                acc_y = -(dy / distance) * acc_magnitude * 0.01
                
                # Apply acceleration
                particle.vx += acc_x
                particle.vy += acc_y
            
            # Update position
            particle.x += particle.vx
            particle.y += particle.vy
            
            # Update angle and radius
            particle.angle = math.atan2(particle.y, particle.x)
            particle.radius = math.sqrt(particle.x**2 + particle.y**2)
            
            # Remove particles that fall into the black hole
            if particle.radius < self.radius:
                self.accretion_particles.remove(particle)
                # Add energy to jets
                self.add_jet_particle()
        
        # Add new particles periodically
        if random.random() < 0.1:
            self.add_accretion_particle()
    
    def add_accretion_particle(self):
        """Add a new particle to the accretion disk"""
        if len(self.accretion_particles) < 80:
            disk_radius = random.uniform(self.radius * 6, self.radius * 10)
            angle = random.uniform(0, 2 * math.pi)
            
            px = math.cos(angle) * disk_radius
            py = math.sin(angle) * disk_radius
            
            orbital_speed = math.sqrt(SIMULATION_G * self.mass / disk_radius) * 0.1
            vx = -math.sin(angle) * orbital_speed
            vy = math.cos(angle) * orbital_speed
            
            heat_factor = max(0.0, 1.0 - (disk_radius - self.radius * 2) / (self.radius * 6))
            if heat_factor > 0.7:
                color = (255, 255, max(0, min(255, int(255 * heat_factor))))
            elif heat_factor > 0.4:
                color = (255, max(0, min(255, int(255 * heat_factor))), 0)
            else:
                color = (max(0, min(255, int(255 * heat_factor))), 0, 0)
            
            particle = Particle(
                x=px, y=py, vx=vx, vy=vy,
                angle=angle, radius=disk_radius,
                color=color, lifetime=0, max_lifetime=random.uniform(300, 600)
            )
            self.accretion_particles.append(particle)
    
    def add_jet_particle(self):
        """Add particles to the relativistic jets"""
        if len(self.jet_particles) < 30:
            # Jets emanate perpendicular to the accretion disk
            for direction in [-1, 1]:  # Two jets, opposite directions
                jet_speed = random.uniform(15, 25)
                angle_offset = random.uniform(-0.2, 0.2)  # Small random spread
                
                vx = 0 + math.cos(angle_offset) * jet_speed * 0.1
                vy = direction * jet_speed + math.sin(angle_offset) * jet_speed * 0.1
                
                color = random.choice([CYAN, BLUE, WHITE])
                
                particle = Particle(
                    x=0, y=0, vx=vx, vy=vy,
                    angle=0, radius=0,
                    color=color, lifetime=0, max_lifetime=random.uniform(60, 120)
                )
                self.jet_particles.append(particle)
    
    def update_jets(self):
        """Update jet particles"""
        for particle in self.jet_particles[:]:
            particle.lifetime += 1
            
            if particle.lifetime > particle.max_lifetime:
                self.jet_particles.remove(particle)
                continue
            
            # Update position
            particle.x += particle.vx
            particle.y += particle.vy
            
            # Fade color over time
            fade_factor = max(0.0, 1.0 - (particle.lifetime / particle.max_lifetime))
            r, g, b = particle.color
            particle.color = (
                max(0, min(255, int(r * fade_factor))),
                max(0, min(255, int(g * fade_factor))),
                max(0, min(255, int(b * fade_factor)))
            )
    
    def update_trail(self):
        """Update position trail"""
        self.trail_points.append((self.x, self.y))
        if len(self.trail_points) > self.max_trail_length:
            self.trail_points.pop(0)
    
    def apply_force(self, fx, fy):
        """Apply gravitational force"""
        ax = fx / self.mass
        ay = fy / self.mass
        self.vx += ax
        self.vy += ay
    
    def update_position(self):
        """Update quasar position"""
        self.x += self.vx
        self.y += self.vy
        self.update_trail()
    
    def draw(self, screen):
        """Draw the quasar and all its components"""
        # Draw trail
        if len(self.trail_points) > 1:
            for i in range(1, len(self.trail_points)):
                alpha = i / len(self.trail_points)
                color = tuple(max(0, min(255, int(c * alpha))) for c in self.color)
                if i < len(self.trail_points) - 1:
                    pygame.draw.line(screen, color, self.trail_points[i-1], self.trail_points[i], 2)
        
        # Draw accretion disk particles
        for particle in self.accretion_particles:
            world_x = self.x + particle.x
            world_y = self.y + particle.y
            if 0 <= world_x < WINDOW_WIDTH and 0 <= world_y < WINDOW_HEIGHT:
                size = max(1, int(4 * (1 - particle.lifetime / particle.max_lifetime)))
                # Ensure color values are valid
                color = tuple(max(0, min(255, int(c))) for c in particle.color)
                pygame.draw.circle(screen, color, (int(world_x), int(world_y)), size)
        
        # Draw jet particles
        for particle in self.jet_particles:
            world_x = self.x + particle.x
            world_y = self.y + particle.y
            if 0 <= world_x < WINDOW_WIDTH and 0 <= world_y < WINDOW_HEIGHT:
                size = max(1, int(3 * (1 - particle.lifetime / particle.max_lifetime)))
                # Ensure color values are valid
                color = tuple(max(0, min(255, int(c))) for c in particle.color)
                pygame.draw.circle(screen, color, (int(world_x), int(world_y)), size)
        
        # Draw event horizon (black hole)
        pygame.draw.circle(screen, BLACK, (int(self.x), int(self.y)), int(self.radius))
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(self.radius), 2)
        
        # Draw name
        font = pygame.font.Font(None, 24)
        text = font.render(self.name, True, self.color)
        screen.blit(text, (self.x - 30, self.y - self.radius - 25))

class QuasarSimulation:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Quasar Collision Simulator")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Create two quasars
        self.quasar1 = Quasar(300, 400, 1000, RED, "Quasar A")
        self.quasar2 = Quasar(900, 400, 1200, BLUE, "Quasar B")
        
        # Set initial velocities for orbital motion
        self.quasar1.vy = 2
        self.quasar2.vy = -1.5
        
        # Simulation parameters
        self.time_step = 0.1
        self.total_time = 0
        self.paused = False
        self.show_vectors = False
        
        # Energy tracking
        self.initial_energy = self.calculate_total_energy()
        
    def calculate_distance(self, q1, q2):
        """Calculate distance between two quasars"""
        dx = q2.x - q1.x
        dy = q2.y - q1.y
        return math.sqrt(dx**2 + dy**2)
    
    def calculate_gravitational_force(self, q1, q2):
        """Calculate gravitational force between two quasars"""
        dx = q2.x - q1.x
        dy = q2.y - q1.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < 1:  # Avoid division by zero
            return 0, 0
        
        # Force magnitude
        force_magnitude = SIMULATION_G * q1.mass * q2.mass / (distance**2)
        
        # Force components
        fx = force_magnitude * (dx / distance)
        fy = force_magnitude * (dy / distance)
        
        return fx, fy
    
    def calculate_total_energy(self):
        """Calculate total energy of the system"""
        # Kinetic energy
        ke1 = 0.5 * self.quasar1.mass * (self.quasar1.vx**2 + self.quasar1.vy**2)
        ke2 = 0.5 * self.quasar2.mass * (self.quasar2.vx**2 + self.quasar2.vy**2)
        
        # Potential energy
        distance = self.calculate_distance(self.quasar1, self.quasar2)
        if distance > 0:
            pe = -SIMULATION_G * self.quasar1.mass * self.quasar2.mass / distance
        else:
            pe = 0
        
        return ke1 + ke2 + pe
    
    def update_physics(self):
        """Update physics simulation"""
        if self.paused:
            return
        
        # Calculate gravitational forces
        fx, fy = self.calculate_gravitational_force(self.quasar1, self.quasar2)
        
        # Apply forces (Newton's 3rd law)
        self.quasar1.apply_force(fx * self.time_step, fy * self.time_step)
        self.quasar2.apply_force(-fx * self.time_step, -fy * self.time_step)
        
        # Update positions
        self.quasar1.update_position()
        self.quasar2.update_position()
        
        # Update accretion disks and jets
        self.quasar1.update_accretion_disk(self.quasar2)
        self.quasar2.update_accretion_disk(self.quasar1)
        
        self.quasar1.update_jets()
        self.quasar2.update_jets()
        
        # Add jet particles periodically
        if random.random() < 0.1:
            self.quasar1.add_jet_particle()
        if random.random() < 0.1:
            self.quasar2.add_jet_particle()
        
        # Check for collision
        distance = self.calculate_distance(self.quasar1, self.quasar2)
        if distance < (self.quasar1.radius + self.quasar2.radius):
            self.handle_collision()
        
        self.total_time += self.time_step
    
    def handle_collision(self):
        """Handle collision between quasars"""
        # Calculate center of mass
        total_mass = self.quasar1.mass + self.quasar2.mass
        cm_x = (self.quasar1.x * self.quasar1.mass + self.quasar2.x * self.quasar2.mass) / total_mass
        cm_y = (self.quasar1.y * self.quasar1.mass + self.quasar2.y * self.quasar2.mass) / total_mass
        
        # Create merged quasar
        merged_mass = total_mass
        merged_color = (
            (self.quasar1.color[0] + self.quasar2.color[0]) // 2,
            (self.quasar1.color[1] + self.quasar2.color[1]) // 2,
            (self.quasar1.color[2] + self.quasar2.color[2]) // 2
        )
        
        # Conservation of momentum
        total_momentum_x = self.quasar1.mass * self.quasar1.vx + self.quasar2.mass * self.quasar2.vx
        total_momentum_y = self.quasar1.mass * self.quasar1.vy + self.quasar2.mass * self.quasar2.vy
        
        merged_vx = total_momentum_x / merged_mass
        merged_vy = total_momentum_y / merged_mass
        
        # Replace quasars with merged one
        self.quasar1 = Quasar(cm_x, cm_y, merged_mass, merged_color, "Merged Quasar")
        self.quasar1.vx = merged_vx
        self.quasar1.vy = merged_vy
        
        # Remove second quasar by making it invisible
        self.quasar2.x = -1000
        self.quasar2.y = -1000
        self.quasar2.vx = 0
        self.quasar2.vy = 0
    
    def draw_ui(self):
        """Draw user interface"""
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)
        
        # Title
        title = font.render("Quasar Collision Simulator", True, CYAN)
        self.screen.blit(title, (10, 10))
        
        # Instructions
        instructions = [
            "SPACE: Pause/Resume",
            "V: Toggle velocity vectors",
            "R: Reset simulation",
            "ESC: Exit",
            "",
            f"Time: {self.total_time:.1f}s",
            f"Distance: {self.calculate_distance(self.quasar1, self.quasar2):.1f}",
            f"Energy: {self.calculate_total_energy():.1f}",
        ]
        
        for i, instruction in enumerate(instructions):
            color = WHITE if instruction else CYAN
            text = small_font.render(instruction, True, color)
            self.screen.blit(text, (10, 60 + i * 20))
        
        # Status
        status = "PAUSED" if self.paused else "RUNNING"
        status_color = YELLOW if self.paused else GREEN
        status_text = small_font.render(status, True, status_color)
        self.screen.blit(status_text, (WINDOW_WIDTH - 100, 10))
    
    def draw_velocity_vectors(self):
        """Draw velocity vectors"""
        if not self.show_vectors:
            return
        
        scale = 20  # Scale factor for visibility
        
        # Quasar 1 velocity vector
        start_pos = (int(self.quasar1.x), int(self.quasar1.y))
        end_pos = (int(self.quasar1.x + self.quasar1.vx * scale), 
                   int(self.quasar1.y + self.quasar1.vy * scale))
        pygame.draw.line(self.screen, self.quasar1.color, start_pos, end_pos, 3)
        
        # Quasar 2 velocity vector
        start_pos = (int(self.quasar2.x), int(self.quasar2.y))
        end_pos = (int(self.quasar2.x + self.quasar2.vx * scale), 
                   int(self.quasar2.y + self.quasar2.vy * scale))
        pygame.draw.line(self.screen, self.quasar2.color, start_pos, end_pos, 3)
    
    def reset_simulation(self):
        """Reset the simulation"""
        self.quasar1 = Quasar(300, 400, 1000, RED, "Quasar A")
        self.quasar2 = Quasar(900, 400, 1200, BLUE, "Quasar B")
        self.quasar1.vy = 2
        self.quasar2.vy = -1.5
        self.total_time = 0
        self.initial_energy = self.calculate_total_energy()
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_v:
                    self.show_vectors = not self.show_vectors
                elif event.key == pygame.K_r:
                    self.reset_simulation()
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def run(self):
        """Main simulation loop"""
        while self.running:
            self.handle_events()
            self.update_physics()
            
            # Draw everything
            self.screen.fill(BLACK)
            
            # Draw stars background
            for _ in range(100):
                x = random.randint(0, WINDOW_WIDTH)
                y = random.randint(0, WINDOW_HEIGHT)
                brightness = random.randint(50, 150)
                pygame.draw.circle(self.screen, (brightness, brightness, brightness), (x, y), 1)
            
            # Draw quasars
            self.quasar1.draw(self.screen)
            if self.quasar2.x > 0:  # Only draw if not merged
                self.quasar2.draw(self.screen)
            
            # Draw velocity vectors
            self.draw_velocity_vectors()
            
            # Draw UI
            self.draw_ui()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()

if __name__ == "__main__":
    sim = QuasarSimulation()
    sim.run()