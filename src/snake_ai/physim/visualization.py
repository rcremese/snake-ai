from snake_ai.physim.simulation import Simulation
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from phi.jax import flow
import numpy as np

def plot_concentration_with_gradient(field : flow.CenteredGrid, output : Optional[str] = None):
    gradient = flow.field.spatial_gradient(field)
    fig, ax = plt.subplots(1,2, dpi=300)
    ax[0].imshow(field.values.numpy('y,x'), cmap='viridis')
    ax[0].set(title='Concentration map & gradients', xlabel='x', ylabel='y')
    np_grad = gradient.values.numpy('vector,y,x')
    ax[0].quiver(np_grad[0], np_grad[1], angles='xy', scale_units='xy', scale=1)
    ax[1].imshow(np.linalg.norm(np_grad, axis=0))
    ax[1].set(title='Gradient norm', xlabel='x', ylabel='y')

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)

def animate_walker_history(field : flow.CenteredGrid, history : List[flow.Tensor], output : Optional[str] = None):
    fig, ax = plt.subplots()
    ax.imshow(field.values.numpy('y,x'))
    ax.set(xlabel='x', ylabel='y', title='Concentration map + deterministic walkers')
    anim = []
    for point_cloud in history:
        points = point_cloud.numpy('point,vector')
        # anim.append(flow.plot(point_cloud, animate=True))
        anim.append(ax.plot(points[:,0], points[:,1], marker="o", ls="", animated=True, color='orange'))
    anim = animation.ArtistAnimation(fig, anim, blit=True, interval=200)
    if output is not None:
        output = Path(output).resolve()
        anim.save(output)

def animate_simulation_history(simulation : Simulation, output : Optional[str] = None):
    fps = 10
    fig, ax = plt.subplots(dpi=300)
    history = simulation.history
    ax.imshow(history[0].values.numpy('y,x'), cmap='inferno')
    ax.set(xlabel='x', ylabel='y', title='Concentration map')

    def animate(i):
        ax.clear()
        im = ax.imshow(history[i].values.numpy('y,x'), cmap='inferno')
        return [im]
    anim = animation.FuncAnimation(fig, animate, frames=len(history), interval=1000/fps)
    anim.save(output)

def animate_walk_history(concentration : flow.CenteredGrid, history : List[flow.PointCloud], output : Optional[str] = None):
    fig, ax = plt.subplots(dpi=300)
    ax.imshow(concentration.values.numpy('y,x'))
    ax.set(xlabel='x', ylabel='y', title='Concentration map + deterministic walkers')
    anim = []
    for point_cloud in history:
        points = point_cloud.numpy('point,vector')
        # anim.append(flow.plot(point_cloud, animate=True))
        anim.append(ax.plot(points[:,0], points[:,1], marker="o", ls="", animated=True, color='orange'))
    anim = animation.ArtistAnimation(fig, anim, blit=True, interval=200)
    anim.save(output)