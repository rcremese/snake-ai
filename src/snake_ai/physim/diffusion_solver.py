from phi.jax import flow
from phi.physics.fluid import apply_boundary_conditions, Obstacle
import matplotlib.pyplot as plt

bounds = flow.Box(x=10, y=10)
obstacle = Obstacle(flow.Box(x=(0, 4), y=(0, 4)))  

initial_distrib = flow.Sphere(x=5, y=5, radius=2)
concentration = 100 * flow.CenteredGrid(initial_distrib, bounds=bounds, resolution=100)

D = 1
dt = 0.5 * concentration.dx ** 2 / D 

@flow.math.jit_compile
def step(concentration, obstacle_field : flow.GeometryMask):
    temp =  flow.diffuse.explicit(concentration, diffusivity=D, dt=dt, substeps=10)
    return obstacle_field * temp

@flow.math.jit_compile
def new_step(velocity, obstacle):
    temp =  flow.diffuse.explicit(velocity, diffusivity=D, dt=dt, substeps=10)
    return apply_boundary_conditions(temp, obstacle)    
T_max = 200
i_print = T_max // 5

trajectory = [apply_boundary_conditions(concentration, [obstacle])]
for i in range(T_max):
    concentration = new_step(concentration, [obstacle])
    if i % i_print == 0:
        trajectory.append(concentration)
trajectory = flow.field.stack(trajectory, flow.batch('time'))
fig = flow.vis.plot(trajectory, show_color_bar=False)
plt.show()