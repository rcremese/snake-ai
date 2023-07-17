from snake_ai.physim.simulation import Simulation
from typing import List, Optional
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from phi.jax import flow
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_concentration_map(
    field: flow.CenteredGrid,
    force_field: Optional[flow.CenteredGrid] = None,
    cmap: str = "inferno",
    title: str = "Concentration map",
):
    fig, ax = plt.subplots(1, 1, dpi=300)
    # spatial information
    x_max, y_max = int(field.bounds.upper["x"]), int(field.bounds.upper["y"])
    # Plot the concentration map
    concentration = field.values.numpy("y,x")
    im = ax.imshow(
        concentration, origin="lower", cmap=cmap, extent=[0, x_max, 0, y_max]
    )
    # Plot the force field if any
    if force_field is not None:
        gradient = force_field.values.numpy("vector, y, x")
        ax.quiver(
            np.linspace(0, y_max, int(field.resolution["y"])),
            np.linspace(0, x_max, int(field.resolution["x"])),
            gradient[0],
            gradient[1],
            angles="xy",
            scale_units="xy",
            scale=1,
        )

    ax.set(title=title, xlabel="x", ylabel="y")
    fig.colorbar(im, ax=ax)
    return fig, ax, im


def plot_walkers_with_concentration(
    point_cloud: flow.PointCloud,
    concentration: Optional[flow.CenteredGrid] = None,
    target: Optional[flow.Tensor] = None,
    force_field: Optional[flow.CenteredGrid] = None,
    cmap_concentration: str = "inferno",
    cmap_walkers: str = "afmhot",
    title: str = "Concentration map + walkers",
):
    assert isinstance(point_cloud, flow.PointCloud)
    assert concentration is None or isinstance(concentration, flow.CenteredGrid)
    assert target is None or isinstance(target, flow.Tensor)
    assert force_field is None or isinstance(force_field, flow.CenteredGrid)

    positions = point_cloud.points.numpy("walker,vector")
    # Create colormaps
    colormap = matplotlib.colormaps[cmap_walkers]
    cloud_colors = colormap(np.linspace(0, 1, positions.shape[0]))

    # Plot the concentration map as backgroud if it exists
    if concentration is None:
        fig, ax = plt.subplots(dpi=300)
    else:
        fig, ax, _ = plot_concentration_map(
            concentration, cmap=cmap_concentration, title=title
        )

    # Plot the force field at walkers positions
    if force_field is not None:
        sampled_field = force_field.at(point_cloud)
        gradient = sampled_field.values.numpy("walker,vector")
        ax.quiver(
            positions[:, 0],
            positions[:, 1],
            gradient[:, 0],
            gradient[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
        )

    # Print the target if it exists
    if target is not None:
        tx, ty = target.numpy("vector")
        ax.plot(tx, ty, color="green", marker="x", markersize=10, ls="", label="Target")
        fig.legend()

    # Plot the walkers with appropriate colors
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c=cloud_colors)
    return fig, ax, scatter


def animate_walk_history(
    trajectories: flow.PointCloud,
    concentration: flow.CenteredGrid,
    output: str,
    target: Optional[flow.Tensor] = None,
    force_field: Optional[flow.CenteredGrid] = None,
    cmap_concentration: str = "inferno",
    cmap_walkers: str = "afmhot",
    title: str = "Concentration map + walkers",
    time_step: int = 1,
):
    # snapshots = [point_cloud.numpy('point,vector') for point_cloud in history]
    snapshots = trajectories.points.numpy("time,walker,vector")
    # Create colormap
    fig, ax, scatter = plot_walkers_with_concentration(
        trajectories.time[0],
        concentration,
        target,
        force_field,
        cmap_concentration,
        cmap_walkers,
        title,
    )

    def update(frame):
        # read new points
        ax.set_title(f"{title} - t={frame * time_step}")
        scatter.set_offsets(snapshots[frame])

    anim = animation.FuncAnimation(
        fig, update, frames=range(snapshots.shape[0]), interval=200
    )
    anim.save(output, fps=10)
    plt.close(fig)


def plot_loss(loss: List[float], output: Optional[str] = None):
    fig, ax = plt.subplots(1, 1, dpi=300)
    ax.plot(loss)
    ax.set(title="Loss function evolution", xlabel="step", ylabel="Loss", yscale="log")
    if output is not None:
        fig.savefig(output)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    from snake_ai.utils.io import SimulationLoader
    import matplotlib.pyplot as plt
    from snake_ai.physim.autodiff import compute_log_concentration, clip_gradient_norm
    import time

    dirpath = Path("/home/rocremes/projects/snake-ai/simulations")
    simulation_path = dirpath.joinpath("GridWorld(20,20)_meta_Tmax=400.0_D=1/seed_0")
    simulation_path = dirpath.joinpath(
        "RandomObstacles(20,20)_pixel_Tmax=400.0_D=1/seed_0"
    )
    simulation_path = dirpath.joinpath("Slot(20,20)_pixel_Tmax=400.0_D=1/seed_0")

    loader = SimulationLoader(simulation_path)
    simu = loader.load()
    # concentration = compute_log_concentration(simu.field)
    concentration = simu.field
    force_field = flow.field.spatial_gradient(concentration, type=flow.CenteredGrid)
    # Plot the concentration field and the walkers inital state
    pt_cloud = simu.point_cloud

    fig, ax, im = plot_concentration_map(
        concentration, force_field, title="Concentration map"
    )
    fig, ax, scatter = plot_walkers_with_concentration(
        pt_cloud,
        concentration,
        force_field=force_field,
        title="Concentration map + walkers",
    )
    fig = flow.vis.plot(concentration, title="Concentration map")
    fig.savefig(dirpath.joinpath("test.png"))
