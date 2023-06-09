from snake_ai.physim.simulation import Simulation
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from phi.jax import flow
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_concentration_with_gradient(field : flow.CenteredGrid, output : Optional[str] = None):
    gradient = flow.field.spatial_gradient(field)
    np_grad = gradient.values.numpy('vector,y,x')

    fig, ax = plt.subplots(1,2, dpi=300)
    im1 = ax[0].imshow(field.values.numpy('y,x'), cmap='viridis')
    ax[0].set(title='Concentration map & gradients', xlabel='x', ylabel='y')
    ax[0].quiver(np_grad[0], np_grad[1], angles='xy', scale_units='xy', scale=1)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax)

    im2 = ax[1].imshow(np.linalg.norm(np_grad, axis=0))
    ax[1].set(title='Gradient norm', xlabel='x', ylabel='y')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)

def animate_walk_history(concentration : flow.CenteredGrid, history : List[flow.PointCloud], output : Optional[str] = None):
    snapshots = [point_cloud.numpy('point,vector') for point_cloud in history]

    fig, ax = plt.subplots(dpi=300)
    ax.imshow(concentration.values.numpy('y,x'))
    ax.set(xlabel='x', ylabel='y', title='Concentration map + deterministic walkers')
    anim = []
    for points in snapshots:
        # anim.append(flow.plot(point_cloud, animate=True))
        # ax.plot(points[0,0], points[0,1], marker="o", ls="", animated=True, color='orange')
        # im = ax.plot(points[1,0], points[1,1], marker="o", ls="", animated=True, color='blue')
        anim.append(ax.plot(points[:,0], points[:,1], marker="o", ls="", animated=True, color='orange'))
    anim = animation.ArtistAnimation(fig, anim, blit=True, interval=200)
    anim.save(output, fps=10)

def animate_simulation_history(history : List[flow.CenteredGrid], output : Optional[str] = None):
    snapshots = [field.values.numpy('y,x') for field in history]

   # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(8,8), dpi=300 )

    im = plt.imshow(snapshots[0], interpolation='none', aspect='auto', vmin=0, vmax=1)

    def animate_func(i):
        im.set_array(snapshots[i])
        return [im]
    fps = 10
    anim = animation.FuncAnimation(
                                fig,
                                animate_func,
                                frames = len(snapshots),
                                interval = 1000 / fps, # in ms
                                )

    anim.save(output, fps=fps)

if __name__ == "__main__":
    from snake_ai.envs import MazeGrid, RandomObstaclesEnv, SlotEnv, RoomEscape
    from snake_ai.physim.converter import DiffusionConverter, ObstacleConverter
    import matplotlib.pyplot as plt
    import time

    env = RandomObstaclesEnv(render_mode="human", seed=0, nb_obs=10, max_obs_size=3)
    env.reset()
    env.render()
    time.sleep(20)
    # converter = DiffusionConverter("pixel")
    # obs_converter = ObstacleConverter("pixel")
    # field = converter(env)
    # obstacles = obs_converter(env)
    # solver = DiffusionSolver(1, 1000, 0.1, endless=True, history_step=10)
    # concentration = solver.solve(field, obstacles)
    # env.render()
    # plot_concentration(solver.history)
