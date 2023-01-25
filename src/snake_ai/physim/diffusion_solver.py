from phi.jax import flow
import matplotlib.pyplot as plt


bounds = flow.Box(x=10, y=10)
obstacle = flow.Obstacle(flow.Box(x=(5, 6), y=(5, 6))) 

concentration = 100 * flow.CenteredGrid(flow.Sphere(center=flow.vec(x=2, y=2), radius=2), bounds=bounds, resolution=100)

D = 1
dt = 0.5 * concentration.dx ** 2 / D 

@flow.math.jit_compile
def step(concentration, obstacle : flow.Obstacle):
    temp =  flow.diffuse.explicit(concentration, diffusivity=D, dt=dt, substeps=10)
    return obstacle @ temp

trajectory = [concentration]
for i in range(200):
    concentration = step(concentration, obstacle)
    if i % 20 == 0:
        trajectory.append(concentration)
trajectory = flow.field.stack(trajectory, flow.batch('time'))
fig = flow.vis.plot(trajectory, show_color_bar=False)
plt.show()