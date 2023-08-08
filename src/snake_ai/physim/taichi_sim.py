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
    width : int
    height : int
    
@ti.dataclass
class State:
    pos : tm.vec2
    vel : tm.vec2

@ti.data_oriented
class ParticleSimulation:
    def __init__(self, particles : Particle, force_field : ti.Field, t_max : float, dt : float, bounds : Rectangle, target : tm.vec2) -> None:
        ## TODO : type check
        self.particles = particles
        self._nb_particles = particles.shape[0]
        
        # self._nb_obstacles = len(obstacles)
        # if self._nb_obstacles > 0:
        #     self.obstacles = Rectangle.field(shape=(len(obstacles,)))
        self.timeline = np.linspace(0, t_max, int(t_max / dt), dtype=np.float32)        
        # self.timeline = ti.field(float, shape=timeline.shape)
        # self.timeline.from_numpy(timeline) 
        self.width = bounds.width  
        self.height = bounds.height
        self.trajectories = ti.Vector.field(n=2, dtype=float, shape=(self._nb_particles, len(self.timeline)))
        # ti.Vector.field(2, dtype=ti.f32, shape=(self._nb_particles, int(t_max / dt)))
        
        self.force_field = force_field
        self.t_max = t_max
        self.dt = dt
        
        self.target = target
        self.loss = ti.field(ti.f32, shape=(), needs_grad=True)
    
    @ti.kernel
    def loss_update(self):
        for n in self.particles:
            self.loss[None] += tm.length(self.particles[n].pos - self.target) / self.particles.shape[0]
        # self.loss[None] /= self.particles.shape[0]

    @ti.kernel
    def update(self, dt: float, t_iter : int):
        for n in self.particles:
            i, j =  ti.floor(self.particles[n].pos).cast(int)

            self.particles[n].vel += dt * self.force_field[i,j]
            self.particles[n].pos += self.particles[n].vel * dt
            # self.particles[n].pos += self.force_field[i,j] * dt
            # Collision check
             # x axis
            if self.particles[n].pos[0] < 0 or self.particles[n].pos[0] > self.width:
                self.particles[n].vel[0] = -self.particles[n].vel[0]
            if self.particles[n].pos[0] < 0:
                self.particles[n].pos[0] *= -1
            if self.particles[n].pos[0] > self.width:
                self.particles[n].pos[0] = 2 * self.width - self.particles[n].pos[0]
            # y axis
            if self.particles[n].pos[1] < 0 or self.particles[n].pos[1] > self.height:
                self.particles[n].vel[1] = -self.particles[n].vel[1]
            if self.particles[n].pos[1] < 0:
                self.particles[n].pos[1] *= -1
            if self.particles[n].pos[1] > self.height:
                self.particles[n].pos[1] = 2 * self.height - self.particles[n].pos[1]
                self.particles[n].pos +=  dt * self.force_field[i,j]
            self.trajectories[n, t_iter] = self.particles[n].pos
        
    @ti.kernel
    def _update_trajectories(self, time : int):
        for n in self.particles:
            self.trajectories[time, n] = self.particles[n].pos
            
    def run(self):
        for t_iter, t in enumerate(self.timeline):
            self.update(self.dt, t_iter)
            # self._update_trajectories(i)

@ti.data_oriented
class DifferentiableSimulation:
    def __init__(self, particles : Particle, force_field : ti.Field, t_max : float, dt : float, bounds : Rectangle, target : tm.vec2, max_epoch : int, lr : float) -> None:
        ## TODO : type check
        self._particles = particles
        self.nb_particles = particles.shape[0]
        
        self.nb_steps = int(t_max / dt)
        self._states = State.field(shape=(self.nb_particles, self.nb_steps), needs_grad=True)

        self.bouds = bounds
        self.width = bounds.width  
        self.height = bounds.height
        # ti.Vector.field(2, dtype=ti.f32, shape=(self._nb_particles, int(t_max / dt)))
        
        self.force_field = force_field
        self.t_max = t_max
        self.dt = dt
        
        self.target = target
        self.loss = ti.field(ti.f32, shape=(), needs_grad=True)
        # training parameters
        self.max_epoch = max_epoch
        self.lr = lr
    
    @ti.kernel
    def reset(self):
        for n in self._particles:
            self._states[n, 0].pos = self._particles[n].pos
            self._states[n, 0].vel = self._particles[n].vel
    
    @ti.kernel
    def step(self, t : int):
        for n in range(self.nb_particles):
            i, j =  ti.floor(self._states[n, t - 1].pos).cast(int)
            
            # self._states[n, t].vel = self._states[n, t - 1].vel + self.dt * self.force_field[i,j]
            # self._states[n, t].pos = self._states[n, t - 1].pos + self.dt * self._states[n, t].vel
            self._states[n, t].pos = self._states[n, t - 1].pos + self.dt * self.force_field[i,j]

    def run(self):
        self.reset()
        for t in range(1, self.nb_steps):
            self.step(t)
        self.compute_loss(t)

    @ti.kernel
    def compute_loss(self, t: int):
        for n in range(self.nb_particles):
            self.loss[None] += tm.length(self._states[n, t].pos - self.target) / self.nb_particles
        
    def optimize(self):
        for iter in range(self.max_epoch):
            with ti.ad.Tape(self.loss):
                self.run()
                
            print('Iter=', iter, 'Loss=', self.loss[None])
            self._update_force_field()

    @ti.kernel
    def _update_force_field(self):
        for i, j in self.force_field:
            self.force_field[i,j] -= self.lr * self.force_field.grad[i,j]

        
    @property
    def particles(self):
        return self._particles.to_numpy()
    
    @property
    def trajectories(self):
        return self._states.to_numpy()
    
def main():
    ti.init()

    sim_dir = Path("/home/rcremese/projects/snake-ai/simulations").resolve(strict=True)
    field_path = sim_dir.joinpath("RandomObstacles(20,20)_pixel_Tmax=800.0_D=1", "seed_0", "field.npz")
    with np.load(field_path) as data:
        field = data["data"]
        width, height = field.shape
    field = np.where(field < 1e-6, 1e-6, field)
    field = np.log(field)

    concentration = ti.field(ti.f32, shape=(width, height))
    concentration.from_numpy(field)

    force_field = ti.Vector.field(2, dtype=ti.f32, shape=(width, height), needs_grad=True)

    @ti.kernel
    def fill_force_field():
        for i, j in concentration:
            force_field[i, j] = ti.Vector([0.0, 0.0])
            if i == 0:
                force_field[i, j] = ti.Vector([1.0, 0.0])              
            elif i == width - 1:
                force_field[i, j] = ti.Vector([-1. , 0.])
            elif  j == 0:
                force_field[i, j] = ti.Vector([0.0, 1.0])
            elif j == height - 1:
                force_field[i, j] = ti.Vector([0.0, -1.0])
            else:
                force_field[i, j] = ti.Vector([concentration[i + 1, j] - concentration[i - 1, j],  concentration[i, j + 1] - concentration[i, j - 1] ])
            # normalize the vector field
            if tm.length(force_field[i, j]) > 1:
                force_field[i, j] = tm.normalize(force_field[i, j])

    fill_force_field()    

    nb_part = 100
    # Declares a struct comprising three vectors and one floating-point number
    point_cloud = Particle.field(shape=(nb_part,), needs_grad=True)


    @ti.kernel
    def fill_particle_field():
        for n in point_cloud:
            point_cloud[n].pos = ti.Vector([ti.random() * width, ti.random() * height])
            point_cloud[n].vel = ti.Vector([0.0, 0.0])
            point_cloud[n].radius = 1.0
            
    fill_particle_field()

            
    init_point_cloud = point_cloud.to_numpy()
    bounds = Rectangle(tm.vec2([0.0, 0.0]), width, height)
    
    target = tm.vec2([width / 2, height / 2])
    # simulation = ParticleSimulation(point_cloud, force_field, 1000, 1, bounds, target)
    simulation = DifferentiableSimulation(point_cloud, force_field, t_max=100, dt=1, bounds=bounds, target=target, max_epoch=200, lr=0.1)
    simulation.optimize()

    trajectories = simulation.trajectories
    # simulation(0.1, 200)
    # last_point_cloud = simulation.particles
    # trajectories = simulation.trajectories.to_numpy()
    # Creates a 
    # GUI of the size of the gray-scale image
    plt.imshow(concentration.to_numpy())#, extent=[0, width, 0, height])
    plt.quiver(force_field.to_numpy()[:, :, 1], force_field.to_numpy()[:, :, 0],angles="xy",
                scale_units="xy",
                scale=1,)
    plt.scatter(trajectories['pos'][:,0, 1], trajectories['pos'][:, 0, 0], c="b")
    plt.scatter(trajectories['pos'][:,-1, 1], trajectories['pos'][:, -1,  0], c="r")
    plt.scatter(target[1], target[0], c="k", marker="x")
    plt.show()

if __name__ == "__main__":
    main()