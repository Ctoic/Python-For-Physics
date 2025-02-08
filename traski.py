from vpython import *

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
dt = 1000  # Time step (s)

# Create the Sun
sun = sphere(pos=vector(0, 0, 0), radius=7e9, color=color.yellow, mass=1.989e30)

# Create a planet (e.g., Earth)
planet = sphere(pos=vector(1.496e11, 0, 0), radius=3e9, color=color.blue, mass=5.972e24)
planet.velocity = vector(0, 29780, 0)  # Initial velocity (m/s)

# Create a trail for the planet's orbit
planet.trail = curve(color=color.blue)
# Create Mars
mars = sphere(pos=vector(2.279e11, 0, 0), radius=2e9, color=color.red, mass=6.39e23)
mars.velocity = vector(0, 24130, 0)
mars.trail = curve(color=color.red)

# Update the animation loop
while True:
    rate(100)

    # Update planet (Earth)
    r_planet = planet.pos - sun.pos
    r_planet_mag = mag(r_planet)
    force_planet = -G * sun.mass * planet.mass / r_planet_mag**2 * norm(r_planet)
    planet.velocity += force_planet / planet.mass * dt
    planet.pos += planet.velocity * dt
    planet.trail.append(pos=planet.pos)

    # Update Mars
    r_mars = mars.pos - sun.pos
    r_mars_mag = mag(r_mars)
    force_mars = -G * sun.mass * mars.mass / r_mars_mag**2 * norm(r_mars)
    mars.velocity += force_mars / mars.mass * dt
    mars.pos += mars.velocity * dt
    mars.trail.append(pos=mars.pos)