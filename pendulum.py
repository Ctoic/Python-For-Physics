"""
Projectile Motion with Drag Simulator
A comprehensive physics simulation comparing ideal projectile motion
with real-world air resistance effects.
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mfig  # <-- OK if you need it
from typing import List, Tuple, Optional
import math
import time
from dataclasses import dataclass
from enum import Enum



class IntegrationMethod(Enum):
    """Different numerical integration methods"""
    EULER = "Euler"
    RK4 = "Runge-Kutta 4th Order"
    VERLET = "Verlet"


@dataclass
class ProjectileState:
    """State of a projectile at a given time"""
    x: float
    y: float
    vx: float
    vy: float
    t: float


@dataclass
class Environment:
    """Environmental parameters"""
    gravity: float = 9.81  # m/s²
    air_density: float = 1.225  # kg/m³ (sea level)
    wind_x: float = 0.0  # m/s (horizontal wind)
    wind_y: float = 0.0  # m/s (vertical wind)
    temperature: float = 20.0  # °C
    pressure: float = 101325  # Pa


class Projectile:
    """Represents a projectile with physical properties"""
    
    def __init__(self, mass: float, diameter: float, drag_coefficient: float, 
                 launch_angle: float, launch_speed: float, launch_height: float = 0):
        self.mass = mass  # kg
        self.diameter = diameter  # m
        self.drag_coefficient = drag_coefficient  # dimensionless
        self.cross_sectional_area = math.pi * (diameter / 2) ** 2  # m²
        
        # Launch parameters
        self.launch_angle = math.radians(launch_angle)  # Convert to radians
        self.launch_speed = launch_speed  # m/s
        self.launch_height = launch_height  # m
        
        # Initial conditions
        self.initial_vx = launch_speed * math.cos(self.launch_angle)
        self.initial_vy = launch_speed * math.sin(self.launch_angle)
        
        # Trajectory storage
        self.trajectory_ideal = []
        self.trajectory_drag = []
        self.trajectory_times = []
        
        # Current state
        self.state = ProjectileState(0, launch_height, self.initial_vx, self.initial_vy, 0)
        
    def reset(self):
        """Reset projectile to initial state"""
        self.state = ProjectileState(0, self.launch_height, self.initial_vx, self.initial_vy, 0)
        self.trajectory_ideal.clear()
        self.trajectory_drag.clear()
        self.trajectory_times.clear()


class PhysicsEngine:
    """Physics engine for projectile motion calculations"""
    
    def __init__(self, environment: Environment):
        self.env = environment
        
    def drag_force(self, velocity: Tuple[float, float], projectile: Projectile) -> Tuple[float, float]:
        """Calculate drag force vector"""
        vx, vy = velocity
        
        # Relative velocity (accounting for wind)
        relative_vx = vx - self.env.wind_x
        relative_vy = vy - self.env.wind_y
        
        # Speed relative to air
        relative_speed = math.sqrt(relative_vx**2 + relative_vy**2)
        
        if relative_speed == 0:
            return 0, 0
        
        # Drag force magnitude: F_drag = 0.5 * ρ * v² * C_d * A
        drag_magnitude = (0.5 * self.env.air_density * relative_speed**2 * 
                         projectile.drag_coefficient * projectile.cross_sectional_area)
        
        # Drag force components (opposite to velocity direction)
        drag_x = -drag_magnitude * (relative_vx / relative_speed)
        drag_y = -drag_magnitude * (relative_vy / relative_speed)
        
        return drag_x, drag_y
    
    def acceleration(self, state: ProjectileState, projectile: Projectile, 
                    include_drag: bool = True) -> Tuple[float, float]:
        """Calculate acceleration components"""
        # Gravitational acceleration
        ax = 0
        ay = -self.env.gravity
        
        if include_drag:
            # Add drag acceleration
            drag_x, drag_y = self.drag_force((state.vx, state.vy), projectile)
            ax += drag_x / projectile.mass
            ay += drag_y / projectile.mass
        
        return ax, ay
    
    def euler_step(self, state: ProjectileState, projectile: Projectile, 
                   dt: float, include_drag: bool = True) -> ProjectileState:
        """Euler integration step"""
        ax, ay = self.acceleration(state, projectile, include_drag)
        
        new_x = state.x + state.vx * dt
        new_y = state.y + state.vy * dt
        new_vx = state.vx + ax * dt
        new_vy = state.vy + ay * dt
        new_t = state.t + dt
        
        return ProjectileState(new_x, new_y, new_vx, new_vy, new_t)
    
    def rk4_step(self, state: ProjectileState, projectile: Projectile, 
                 dt: float, include_drag: bool = True) -> ProjectileState:
        """Runge-Kutta 4th order integration step"""
        # k1
        ax1, ay1 = self.acceleration(state, projectile, include_drag)
        k1_x = state.vx * dt
        k1_y = state.vy * dt
        k1_vx = ax1 * dt
        k1_vy = ay1 * dt
        
        # k2
        temp_state = ProjectileState(
            state.x + k1_x/2, state.y + k1_y/2,
            state.vx + k1_vx/2, state.vy + k1_vy/2,
            state.t + dt/2
        )
        ax2, ay2 = self.acceleration(temp_state, projectile, include_drag)
        k2_x = temp_state.vx * dt
        k2_y = temp_state.vy * dt
        k2_vx = ax2 * dt
        k2_vy = ay2 * dt
        
        # k3
        temp_state = ProjectileState(
            state.x + k2_x/2, state.y + k2_y/2,
            state.vx + k2_vx/2, state.vy + k2_vy/2,
            state.t + dt/2
        )
        ax3, ay3 = self.acceleration(temp_state, projectile, include_drag)
        k3_x = temp_state.vx * dt
        k3_y = temp_state.vy * dt
        k3_vx = ax3 * dt
        k3_vy = ay3 * dt
        
        # k4
        temp_state = ProjectileState(
            state.x + k3_x, state.y + k3_y,
            state.vx + k3_vx, state.vy + k3_vy,
            state.t + dt
        )
        ax4, ay4 = self.acceleration(temp_state, projectile, include_drag)
        k4_x = temp_state.vx * dt
        k4_y = temp_state.vy * dt
        k4_vx = ax4 * dt
        k4_vy = ay4 * dt
        
        # Combine
        new_x = state.x + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        new_y = state.y + (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
        new_vx = state.vx + (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx) / 6
        new_vy = state.vy + (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy) / 6
        new_t = state.t + dt
        
        return ProjectileState(new_x, new_y, new_vx, new_vy, new_t)
    
    def analytical_solution(self, t: float, projectile: Projectile) -> Tuple[float, float]:
        """Analytical solution for projectile motion without drag"""
        x = projectile.initial_vx * t
        y = projectile.launch_height + projectile.initial_vy * t - 0.5 * self.env.gravity * t**2
        return x, y
    
    def simulate_trajectory(self, projectile: Projectile, dt: float = 0.01, 
                           max_time: float = 100, method: IntegrationMethod = IntegrationMethod.RK4):
        """Simulate complete trajectory"""
        projectile.reset()
        
        # Choose integration method
        if method == IntegrationMethod.EULER:
            step_function = self.euler_step
        elif method == IntegrationMethod.RK4:
            step_function = self.rk4_step
        else:
            step_function = self.euler_step  # Default fallback
        
        # Simulate with drag
        state = projectile.state
        while state.y >= 0 and state.t < max_time:
            projectile.trajectory_drag.append((state.x, state.y))
            projectile.trajectory_times.append(state.t)
            
            # Calculate ideal trajectory point
            ideal_x, ideal_y = self.analytical_solution(state.t, projectile)
            projectile.trajectory_ideal.append((ideal_x, ideal_y))
            
            # Next step
            state = step_function(state, projectile, dt, include_drag=True)
        
        # Add final point
        if state.y < 0:
            # Interpolate to ground level
            prev_state = ProjectileState(
                projectile.trajectory_drag[-1][0],
                projectile.trajectory_drag[-1][1],
                0, 0, projectile.trajectory_times[-1]
            )
            t_impact = prev_state.t - prev_state.y / ((state.y - prev_state.y) / dt)
            
            # Final points
            final_drag_x = prev_state.x + (state.x - prev_state.x) * (t_impact - prev_state.t) / dt
            projectile.trajectory_drag.append((final_drag_x, 0))
            
            final_ideal_x, _ = self.analytical_solution(t_impact, projectile)
            projectile.trajectory_ideal.append((final_ideal_x, 0))
            
            projectile.trajectory_times.append(t_impact)


class ProjectileSimulator:
    """Main simulator class with GUI and visualization"""
    
    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Projectile Motion with Drag Simulator")
        self.clock = pygame.time.Clock()
        
        # Initialize components
        self.environment = Environment()
        self.physics_engine = PhysicsEngine(self.environment)
        
        # Default projectile (baseball)
        self.projectile = Projectile(
            mass=0.145,  # kg
            diameter=0.074,  # m
            drag_coefficient=0.47,
            launch_angle=45,  # degrees
            launch_speed=30,  # m/s
            launch_height=1.5  # m
        )
        
        # Simulation parameters
        self.dt = 0.01
        self.scale = 10  # pixels per meter
        self.origin_x = 100
        self.origin_y = height - 100
        
        # UI state
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.input_mode = None
        self.input_text = ""
        
        # Animation
        self.animation_time = 0
        self.animation_speed = 1.0
        self.show_vectors = True
        self.show_trail = True
        
        # Run initial simulation
        self.simulate()
    
    def simulate(self):
        """Run the physics simulation"""
        self.physics_engine.simulate_trajectory(self.projectile, self.dt)
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        screen_x = int(self.origin_x + x * self.scale)
        screen_y = int(self.origin_y - y * self.scale)
        return screen_x, screen_y
    
    def draw_trajectory(self, trajectory: List[Tuple[float, float]], 
                       color: Tuple[int, int, int], width: int = 2):
        """Draw a trajectory on screen"""
        if len(trajectory) < 2:
            return
        
        screen_points = [self.world_to_screen(x, y) for x, y in trajectory]
        
        # Filter points that are on screen
        visible_points = [(x, y) for x, y in screen_points 
                         if 0 <= x <= self.width and 0 <= y <= self.height]
        
        if len(visible_points) >= 2:
            pygame.draw.lines(self.screen, color, False, visible_points, width)
    
    def draw_projectile(self, position: Tuple[float, float], velocity: Tuple[float, float]):
        """Draw the projectile and its velocity vector"""
        screen_x, screen_y = self.world_to_screen(position[0], position[1])
        
        # Draw projectile
        pygame.draw.circle(self.screen, (255, 255, 0), (screen_x, screen_y), 8)
        pygame.draw.circle(self.screen, (200, 200, 0), (screen_x, screen_y), 8, 2)
        
        if self.show_vectors:
            # Draw velocity vector
            vector_scale = 2
            end_x = screen_x + velocity[0] * vector_scale
            end_y = screen_y - velocity[1] * vector_scale
            
            pygame.draw.line(self.screen, (255, 255, 255), 
                           (screen_x, screen_y), (end_x, end_y), 2)
            
            # Draw arrowhead
            angle = math.atan2(-velocity[1], velocity[0])
            arrow_length = 8
            arrow_angle = math.pi / 6
            
            arrow_x1 = end_x - arrow_length * math.cos(angle - arrow_angle)
            arrow_y1 = end_y - arrow_length * math.sin(angle - arrow_angle)
            arrow_x2 = end_x - arrow_length * math.cos(angle + arrow_angle)
            arrow_y2 = end_y - arrow_length * math.sin(angle + arrow_angle)
            
            pygame.draw.line(self.screen, (255, 255, 255), 
                           (end_x, end_y), (arrow_x1, arrow_y1), 2)
            pygame.draw.line(self.screen, (255, 255, 255), 
                           (end_x, end_y), (arrow_x2, arrow_y2), 2)
    
    def draw_grid(self):
        """Draw coordinate grid"""
        # Vertical lines
        for x in range(0, self.width, int(10 * self.scale)):
            pygame.draw.line(self.screen, (50, 50, 50), (x, 0), (x, self.height), 1)
        
        # Horizontal lines
        for y in range(0, self.height, int(10 * self.scale)):
            pygame.draw.line(self.screen, (50, 50, 50), (0, y), (self.width, y), 1)
        
        # Axes
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (self.origin_x, 0), (self.origin_x, self.height), 2)
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (0, self.origin_y), (self.width, self.origin_y), 2)
    
    def draw_ui(self):
        """Draw user interface elements"""
        # Background for UI
        ui_bg = pygame.Surface((350, 600))
        ui_bg.set_alpha(200)
        ui_bg.fill((0, 0, 0))
        self.screen.blit(ui_bg, (self.width - 360, 10))
        
        # Title
        title = self.font.render("Projectile Motion Simulator", True, (255, 255, 255))
        self.screen.blit(title, (self.width - 350, 20))
        
        # Parameters
        y_offset = 60
        params = [
            f"Launch Angle: {math.degrees(self.projectile.launch_angle):.1f}°",
            f"Launch Speed: {self.projectile.launch_speed:.1f} m/s",
            f"Launch Height: {self.projectile.launch_height:.1f} m",
            f"Mass: {self.projectile.mass:.3f} kg",
            f"Diameter: {self.projectile.diameter:.3f} m",
            f"Drag Coefficient: {self.projectile.drag_coefficient:.2f}",
            f"Gravity: {self.environment.gravity:.1f} m/s²",
            f"Air Density: {self.environment.air_density:.3f} kg/m³",
        ]
        
        for param in params:
            text = self.small_font.render(param, True, (255, 255, 255))
            self.screen.blit(text, (self.width - 340, y_offset))
            y_offset += 25
        
        # Trajectory comparison
        y_offset += 20
        comparison_title = self.font.render("Trajectory Comparison", True, (255, 255, 255))
        self.screen.blit(comparison_title, (self.width - 350, y_offset))
        y_offset += 30
        
        if self.projectile.trajectory_ideal and self.projectile.trajectory_drag:
            ideal_range = self.projectile.trajectory_ideal[-1][0]
            drag_range = self.projectile.trajectory_drag[-1][0]
            range_difference = ideal_range - drag_range
            range_reduction = (range_difference / ideal_range) * 100
            
            ideal_time = self.projectile.trajectory_times[-1]
            
            comparison_stats = [
                f"Ideal Range: {ideal_range:.1f} m",
                f"With Drag: {drag_range:.1f} m",
                f"Reduction: {range_reduction:.1f}%",
                f"Flight Time: {ideal_time:.1f} s",
                f"Max Height: {max(y for x, y in self.projectile.trajectory_ideal):.1f} m"
            ]
            
            for stat in comparison_stats:
                text = self.small_font.render(stat, True, (255, 255, 255))
                self.screen.blit(text, (self.width - 340, y_offset))
                y_offset += 20
        
        # Legend
        y_offset += 30
        legend_title = self.font.render("Legend", True, (255, 255, 255))
        self.screen.blit(legend_title, (self.width - 350, y_offset))
        y_offset += 25
        
        # Ideal trajectory
        pygame.draw.line(self.screen, (0, 255, 0), 
                        (self.width - 340, y_offset + 8), (self.width - 310, y_offset + 8), 3)
        ideal_text = self.small_font.render("Ideal (no drag)", True, (255, 255, 255))
        self.screen.blit(ideal_text, (self.width - 300, y_offset))
        y_offset += 25
        
        # Drag trajectory
        pygame.draw.line(self.screen, (255, 100, 100), 
                        (self.width - 340, y_offset + 8), (self.width - 310, y_offset + 8), 3)
        drag_text = self.small_font.render("With drag", True, (255, 255, 255))
        self.screen.blit(drag_text, (self.width - 300, y_offset))
        y_offset += 25
        
        # Controls
        y_offset += 30
        controls_title = self.font.render("Controls", True, (255, 255, 255))
        self.screen.blit(controls_title, (self.width - 350, y_offset))
        y_offset += 25
        
        controls = [
            "A/D: Adjust angle",
            "S/W: Adjust speed",
            "Q/E: Adjust mass",
            "R: Reset",
            "V: Toggle vectors",
            "SPACE: Animate",
            "ESC: Quit"
        ]
        
        for control in controls:
            text = self.small_font.render(control, True, (255, 255, 255))
            self.screen.blit(text, (self.width - 340, y_offset))
            y_offset += 18
    
    def animate_projectile(self):
        """Animate the projectile along its trajectory"""
        if not self.projectile.trajectory_drag:
            return
        
        # Calculate current position based on animation time
        total_time = self.projectile.trajectory_times[-1]
        current_time = (self.animation_time * self.animation_speed) % total_time
        
        # Find the closest trajectory point
        for i, t in enumerate(self.projectile.trajectory_times):
            if t >= current_time:
                if i == 0:
                    pos = self.projectile.trajectory_drag[0]
                    vel = (self.projectile.initial_vx, self.projectile.initial_vy)
                else:
                    # Interpolate between points
                    t_prev = self.projectile.trajectory_times[i-1]
                    t_curr = self.projectile.trajectory_times[i]
                    alpha = (current_time - t_prev) / (t_curr - t_prev)
                    
                    pos_prev = self.projectile.trajectory_drag[i-1]
                    pos_curr = self.projectile.trajectory_drag[i]
                    
                    pos = (
                        pos_prev[0] + alpha * (pos_curr[0] - pos_prev[0]),
                        pos_prev[1] + alpha * (pos_curr[1] - pos_prev[1])
                    )
                    
                    # Approximate velocity
                    vel = (
                        (pos_curr[0] - pos_prev[0]) / (t_curr - t_prev),
                        (pos_curr[1] - pos_prev[1]) / (t_curr - t_prev)
                    )
                
                self.draw_projectile(pos, vel)
                break
    
    def handle_input(self, event):
        """Handle user input"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            elif event.key == pygame.K_r:
                self.projectile.reset()
                self.simulate()
            elif event.key == pygame.K_v:
                self.show_vectors = not self.show_vectors
            elif event.key == pygame.K_SPACE:
                self.animation_time = 0
            elif event.key == pygame.K_a:
                self.projectile.launch_angle = max(0, self.projectile.launch_angle - 0.1)
                self.projectile.initial_vx = self.projectile.launch_speed * math.cos(self.projectile.launch_angle)
                self.projectile.initial_vy = self.projectile.launch_speed * math.sin(self.projectile.launch_angle)
                self.simulate()
            elif event.key == pygame.K_d:
                self.projectile.launch_angle = min(math.pi/2, self.projectile.launch_angle + 0.1)
                self.projectile.initial_vx = self.projectile.launch_speed * math.cos(self.projectile.launch_angle)
                self.projectile.initial_vy = self.projectile.launch_speed * math.sin(self.projectile.launch_angle)
                self.simulate()
            elif event.key == pygame.K_s:
                self.projectile.launch_speed = max(1, self.projectile.launch_speed - 1)
                self.projectile.initial_vx = self.projectile.launch_speed * math.cos(self.projectile.launch_angle)
                self.projectile.initial_vy = self.projectile.launch_speed * math.sin(self.projectile.launch_angle)
                self.simulate()
            elif event.key == pygame.K_w:
                self.projectile.launch_speed = min(100, self.projectile.launch_speed + 1)
                self.projectile.initial_vx = self.projectile.launch_speed * math.cos(self.projectile.launch_angle)
                self.projectile.initial_vy = self.projectile.launch_speed * math.sin(self.projectile.launch_angle)
                self.simulate()
            elif event.key == pygame.K_q:
                self.projectile.mass = max(0.01, self.projectile.mass - 0.01)
                self.simulate()
            elif event.key == pygame.K_e:
                self.projectile.mass = min(10, self.projectile.mass + 0.01)
                self.simulate()
        
        return True
    
    def run(self):
        """Main simulation loop"""
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    running = self.handle_input(event)
            
            # Update animation
            self.animation_time += self.clock.get_time() / 1000.0
            
            # Clear screen
            self.screen.fill((20, 20, 40))
            
            # Draw grid
            self.draw_grid()
            
            # Draw trajectories
            self.draw_trajectory(self.projectile.trajectory_ideal, (0, 255, 0), 3)
            self.draw_trajectory(self.projectile.trajectory_drag, (255, 100, 100), 3)
            
            # Draw animated projectile
            self.animate_projectile()
            
            # Draw UI
            self.draw_ui()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()


def create_comparison_plot():
    """Create a matplotlib comparison plot"""
    # Create different projectiles for comparison
    projectiles = [
        ("No Drag", Projectile(0.145, 0.074, 0.0, 45, 30, 1.5)),
        ("Baseball", Projectile(0.145, 0.074, 0.47, 45, 30, 1.5)),
        ("Bowling Ball", Projectile(7.26, 0.216, 0.47, 45, 30, 1.5)),
        ("Ping Pong Ball", Projectile(0.0027, 0.04, 0.47, 45, 30, 1.5)),
    ]
    
    env = Environment()
    physics = PhysicsEngine(env)
    
    plt.figure(figsize=(12, 8))
    
    # Plot trajectories
    for name, projectile in projectiles:
        physics.simulate_trajectory(projectile, dt=0.01)
        
        if projectile.trajectory_drag:
            x_coords = [pos[0] for pos in projectile.trajectory_drag]
            y_coords = [pos[1] for pos in projectile.trajectory_drag]
            plt.plot(x_coords, y_coords, label=name, linewidth=2)
    
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Height (m)')
    plt.title('Projectile Motion Comparison: Effect of Drag on Different Objects')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(0, 50)
    
    # Add annotations
    plt.annotate('Launch point', xy=(0, 1.5), xytext=(10, 10),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=12, color='red')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Projectile Motion with Drag Simulator")
    print("Choose an option:")
    print("1. Interactive simulation")
    print("2. Comparison plot")
    print("3. Both")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "2":
        create_comparison_plot()
    elif choice == "3":
        create_comparison_plot()
        print("\nStarting interactive simulation...")
        simulator = ProjectileSimulator()
        simulator.run()
    else:
        simulator = ProjectileSimulator()
        simulator.run()