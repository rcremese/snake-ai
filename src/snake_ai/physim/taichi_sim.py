import taichi as ti
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import taichi.math as tm
from typing import List

@ti.func
def lerp(a, b, t):
    return a + (b - a) * t

@ti.dataclass
class Particle:
    pos: ti.math.vec2
    vel: ti.math.vec2
    radius: ti.f32
    
@ti.dataclass
class Rectangle:
    upper_left: ti.math.vec2
    lower_right: ti.math.vec2
    
@ti.data_oriented
class ParticleSimulation:
    def __init__(self, particles : Particle, force_field : ti.Field, t_max : float, dt : float) -> None:
        ## TODO : type check
        self.particles = particles
        self._nb_particles = particles.shape[0]
        
        # self._nb_obstacles = len(obstacles)
        # if self._nb_obstacles > 0:
        #     self.obstacles = Rectangle.field(shape=(len(obstacles,)))
        self.timeline = np.linspace(0, t_max, int(t_max / dt), dtype=np.float32)        
        # self.timeline = ti.field(float, shape=timeline.shape)
        # self.timeline.from_numpy(timeline) 
    
        self.trajectories = ti.Vector.field(2, dtype=ti.f32, shape=(self._nb_particles, int(t_max / dt)))
        
        self.force_field = force_field
        self.t_max = t_max
        self.dt = dt
        
        
    @ti.kernel
    def update(self, dt: float):
        for n in self.particles:
            i, j =  ti.floor(self.particles[n].pos).cast(int)

            self.particles[n].vel += dt * self.force_field[i,j]
            self.particles[n].pos += self.particles[n].vel * dt
            # self.particles[n].pos +=  dt * self.force_field[i,j]
            
            # self.trajectories[n, time] = self.particles[n].pos
                
    # @ti.func
    # def simulate_trajectory(self, n : int):
    #     for t in self.timeline:
    #         self.update(self.dt, n)
    #         self.trajectories[n, t] = self.particles[n].pos
            
    def run(self):
        # timeline = np.linspace(0, self.t_max, int(self.t_max / self.dt), dtype=np.float32)
       
        for t in self.timeline:
            self.update(self.dt)
        
    @ti.kernel
    def _init_particles(self):
        for n in self.particles:
            self.particles[n].pos = ti.Vector([ti.random() * width, ti.random() * height])
            self.particles[n].vel = ti.Vector([0.0, 0.0])
            self.particles[n].radius = 1.0
ti.init()

part = Particle(ti.Vector([1.0, 0.0]), ti.Vector([0.0, 0.0]), 1.0)
last_point_cloud = Particle.field(shape=(10,))
last_point_cloud[0].pos = ti.Vector([1.0, 0.0])

@ti.kernel
def print_point_cloud(pt_cld : ti.template()):
    for i in pt_cld:
        print(last_point_cloud[i].pos, last_point_cloud[i].vel)
        
sim_dir = Path("/home/rcremese/projects/snake-ai/simulations").resolve(strict=True)
field_path = sim_dir.joinpath("RandomObstacles(20,20)_pixel_Tmax=400.0_D=1", "seed_0", "field.npz")
with np.load(field_path) as data:
    field = data["data"]
    width, height = field.shape
field = np.where(field < 1e-6, 1e-6, field)
field = np.log(field)
gradient = np.gradient(field)

# space = np.linspace(0, 20, width, dtype=np.float32)
# X, Y = np.meshgrid(space, space)
# field = X + Y

concentration = ti.field(ti.f32, shape=(width, height))
concentration.from_numpy(field)

force_field = ti.Vector.field(2, dtype=ti.f32, shape=(width, height))

@ti.kernel
def fill_force_field():
    for i, j in concentration:
        if i == 0 or j == 0 or i == width - 1 or j == height - 1:
            force_field[i, j] = ti.Vector([0.0, 0.0])
        else:
            force_field[i, j] = ti.Vector([concentration[i + 1, j] - concentration[i - 1, j],  concentration[i, j + 1] - concentration[i, j - 1] ])

fill_force_field()    

n = 100
# Declares a struct comprising three vectors and one floating-point number
point_cloud = Particle.field(shape=(n,))
# particle = ti.types.struct(
#   pos=ti.math.vec2, vel=ti.math.vec2, mass=float,
# )
# # Declares a 1D field of the struct particle by calling field()
# particle_field = particle.field(shape=(n,))

@ti.kernel
def fill_particle_field():
    for n in point_cloud:
        point_cloud[n].pos = ti.Vector([ti.random() * width, ti.random() * height])
        point_cloud[n].vel = ti.Vector([0.0, 0.0])
        point_cloud[n].radius = 1.0
        # last_point_cloud[n].mass = 1.0
        
fill_particle_field()

# @ti.func
# def step(dt : float):
#     for walker in range(particle_field.shape[0]):
#         i, j =  ti.floor(particle_field[walker].pos).cast(int)
#         # print(i, j, particle_field[walker].pos, particle_field[walker].vel, force_field[i,j])
#         particle_field[walker].vel += dt * force_field[i,j]
#         particle_field[walker].pos += dt * particle_field[walker].vel
#         # Collision check
#         # x axis
#         if point_cloud[n].pos[0] < 0 or point_cloud[n].pos[0] > width:
#             point_cloud[n].vel[0] = -point_cloud[n].vel[0]
#         if point_cloud[n].pos[0] < 0:
#             point_cloud[n].pos[0] *= -1
#         if point_cloud[n].pos[0] > width:
#             point_cloud[n].pos[0] = 2 * width - point_cloud[n].pos[0]
#         # y axis
#         if point_cloud[n].pos[1] < 0 or point_cloud[n].pos[1] > height:
#             point_cloud[n].vel[1] = -point_cloud[n].vel[1]
#         if point_cloud[n].pos[1] < 0:
#             point_cloud[n].pos[1] *= -1
#         if point_cloud[n].pos[1] > height:
#             point_cloud[n].pos[1] = 2 * height - point_cloud[n].pos[1]

# @ti.kernel
# def simulation(dt : float, t_max : int):
#     for t in range(t_max):
#         step(dt)
        
init_point_cloud = point_cloud.to_numpy()
simulation = ParticleSimulation(point_cloud, force_field, 10, 0.1)
simulation.run()
# simulation(0.1, 200)
last_point_cloud = simulation.particles.to_numpy()
# Creates a 
# GUI of the size of the gray-scale image
plt.imshow(concentration.to_numpy())
plt.quiver(force_field.to_numpy()[:, :, 1], force_field.to_numpy()[:, :, 0],angles="xy",
            scale_units="xy",
            scale=1,)
plt.scatter(init_point_cloud['pos'][:, 0], init_point_cloud['pos'][:, 1], c="b")
plt.scatter(last_point_cloud['pos'][:, 0], last_point_cloud['pos'][:, 1], c="r")
plt.show()
# gui = ti.GUI('gray-scale image of random values', (width, height))
# while gui.running:
#     gui.set_image(concentration)
#     gui.show()
