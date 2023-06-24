from snake_ai.physim.simulation import Simulation
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from phi.jax import flow
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_concentration_map(field : flow.CenteredGrid, output : Optional[str] = None, use_log : bool = False):
    fig, ax = plt.subplots(1,1, dpi=100)
    concentration = field.values.numpy('y,x')
    if use_log:
        concentration = np.log(concentration)
    title = 'Log concentration map' if use_log else 'Concentration map'
    im = ax.imshow(concentration, cmap='inferno')
    ax.set(title=title, xlabel='x', ylabel='y')
    fig.colorbar(im, ax=ax)
    if output is not None:
        fig.savefig(output)
        plt.close(fig)
    else:
        plt.show()

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
    if output is not None:
        fig.savefig(output)
        plt.close(fig)
    else:
        plt.show()

def plot_walkers_with_concentration(
        concentration : flow.CenteredGrid,
        point_cloud : flow.PointCloud,
        target : Optional[flow.Tensor] = None,
        force_field : Optional[flow.CenteredGrid] = None,
        cmap_concentration : str = 'inferno',
        cmap_walkers : str = 'afmhot',
        title : str = 'Concentration map + walkers',
        ):
    point_cloud = point_cloud.points.numpy('walker,vector')
    # Create colormaps
    colormap = plt.cm.get_cmap(cmap_walkers)
    cloud_colors = colormap(np.linspace(0, 1, point_cloud.shape[0]))
    # Create figure and background image
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(concentration.values.numpy('y,x'), cmap=cmap_concentration)
    ax.set(xlabel='x', ylabel='y', title=title)
    if force_field is not None:
        gradient = force_field.values.numpy('vector,y,x')
        ax.quiver(gradient[0], gradient[1], angles='xy', scale_units='xy', scale=1)
    # Print the target if it exists
    if target is not None:
        tx, ty = target.numpy('vector')
        ax.plot(tx, ty, color='blue', marker='x', markersize=10, ls='', label='Target')
        fig.legend()
    fig.colorbar(im, ax=ax)
    scatter = ax.scatter(point_cloud[:,0], point_cloud[:,1], c=cloud_colors)
    return fig, ax, scatter

def animate_walk_history(
        concentration : flow.CenteredGrid,
        history : flow.PointCloud,
        output : str,
        target : Optional[flow.Tensor] = None,
        force_field : Optional[flow.CenteredGrid] = None,
        cmap_concentration : str = 'inferno',
        cmap_walkers : str = 'afmhot',
        title : str = 'Concentration map + walkers',
        ):
    # snapshots = [point_cloud.numpy('point,vector') for point_cloud in history]
    snapshots = history.points.numpy('time,walker,vector')
    # Create colormap
    fig, _, scatter = plot_walkers_with_concentration(concentration, history.time[0], target, force_field, cmap_concentration, cmap_walkers, title)

    def update(frame):
        # read new points
        scatter.set_offsets(snapshots[frame])

    anim = animation.FuncAnimation(fig, update, frames=range(snapshots.shape[0]), interval=200)
    anim.save(output, fps=10)
    plt.close(fig)

def plot_loss(loss : List[float], output : Optional[str] = None):
    fig, ax = plt.subplots(1,1, dpi=300)
    ax.plot(loss)
    ax.set(title='Loss function evolution', xlabel='step', ylabel='Loss')
    if output is not None:
        fig.savefig(output)
        plt.close(fig)
    else:
        plt.show()

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
