import taichi as ti
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

ti.init()

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
particle = ti.types.struct(
  pos=ti.math.vec2, vel=ti.math.vec2, mass=float,
)
# Declares a 1D field of the struct particle by calling field()
particle_field = particle.field(shape=(n,))

@ti.kernel
def fill_particle_field():
    for i in range(particle_field.shape[0]):
        particle_field[i].pos = ti.Vector([ti.random() * width, ti.random() * height])
        particle_field[i].vel = ti.Vector([0.0, 0.0])
        particle_field[i].mass = 1.0
        
fill_particle_field()

@ti.func
def step(dt : float):
    for walker in range(particle_field.shape[0]):
        i, j =  ti.floor(particle_field[walker].pos).cast(int)
        # print(i, j, particle_field[walker].pos, particle_field[walker].vel, force_field[i,j])
        particle_field[walker].vel += dt * force_field[i,j]
        particle_field[walker].pos += dt * particle_field[walker].vel
        # Collision check
        # x axis
        if particle_field[i].pos[0] < 0 or particle_field[i].pos[0] > width:
            particle_field[i].vel[0] = -particle_field[i].vel[0]
        if particle_field[i].pos[0] < 0:
            particle_field[i].pos[0] *= -1
        if particle_field[i].pos[0] > width:
            particle_field[i].pos[0] = 2 * width - particle_field[i].pos[0]
        # y axis
        if particle_field[i].pos[1] < 0 or particle_field[i].pos[1] > height:
            particle_field[i].vel[1] = -particle_field[i].vel[1]
        if particle_field[i].pos[1] < 0:
            particle_field[i].pos[1] *= -1
        if particle_field[i].pos[1] > height:
            particle_field[i].pos[1] = 2 * height - particle_field[i].pos[1]

@ti.kernel
def simulation(dt : float, t_max : int):
    for t in range(t_max):
        step(dt)
        
init_point_cloud = particle_field.to_numpy()
simulation(0.1, 200)
point_cloud = particle_field.to_numpy()
# Creates a 
# GUI of the size of the gray-scale image
plt.imshow(concentration.to_numpy())
plt.quiver(force_field.to_numpy()[:, :, 1], force_field.to_numpy()[:, :, 0],angles="xy",
            scale_units="xy",
            scale=1,)
plt.scatter(init_point_cloud['pos'][:, 0], init_point_cloud['pos'][:, 1], c="b")
plt.scatter(point_cloud['pos'][:, 0], point_cloud['pos'][:, 1], c="r")
plt.show()
# gui = ti.GUI('gray-scale image of random values', (width, height))
# while gui.running:
#     gui.set_image(concentration)
#     gui.show()
