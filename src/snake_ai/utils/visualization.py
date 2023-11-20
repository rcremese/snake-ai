import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
import numpy as np

from snake_ai.envs import GridWorld
from snake_ai.envs.geometry import Cube, Rectangle
from snake_ai.taichi.field import SampledField, ScalarField, spatial_gradient
from typing import List, Tuple, Optional


class Visualizer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def plot_2D_field(
        concentration: ScalarField,
        title: str = "Concentration map",
        add_quiver: bool = False,
    ):
        assert isinstance(concentration, ScalarField), "Expected a scalar field"
        fig, ax = plt.subplots(1, 1, dpi=300)

        im = ax.imshow(
            concentration.values.T,
            extent=[
                concentration._bounds.min[0],
                concentration._bounds.max[0],
                concentration._bounds.max[1],
                concentration._bounds.min[1],
            ],
            cmap="inferno",
        )
        if add_quiver:
            force_field = spatial_gradient(concentration)

            X, Y = force_field.meshgrid
            ax.quiver(
                X,
                Y,
                force_field.values[:, :, 0],
                force_field.values[:, :, 1],
                units="xy",
                angles="xy",
                scale=1,
            )
        ax.set(title=title, xlabel="x", ylabel="y")
        fig.colorbar(im, ax=ax)
        return fig, ax

    @staticmethod
    def plot_2D_environment(env: GridWorld):
        assert isinstance(env, GridWorld), "Expected a GridWorld environment"
        fig, ax = plt.subplots(1, 1, dpi=300)
        for obstacle in env.obstacles:
            ax.add_patch(
                plt.Rectangle(
                    (obstacle.x, obstacle.y),
                    obstacle.width,
                    obstacle.height,
                    color="red",
                )
            )
        ax.add_patch(plt.Circle(env.goal.center, 0.5, color="green"))
        ax.add_patch(plt.Circle(env.agent.position.center, 0.5, color="blue"))
        ax.set(title=f"Environment {env.name}", xlabel="x", ylabel="y")
        plt.xlim(0, env.width)
        plt.ylim(0, env.height)
        return fig, ax


def plot_concentration_map(
    field_values: np.ndarray,
    bound_limits: Optional[Tuple[float, float]] = None,
    force_field: Optional[np.ndarray] = None,
    cmap: str = "inferno",
    title: str = "Concentration map",
):
    """_summary_

    Args:
        field_values (np.ndarray): A 2D concentration field. The first dimension is supposed to be the x axis.
        force_field (np.ndarray, optional):A 2D vector field superposable with field. The first dimension is x,y-componants of the force-field. Defaults to None.
        cmap (str, optional): _description_. Defaults to "inferno".
        title (str, optional): _description_. Defaults to "Concentration map".

    Returns:
        _type_: _description_
    """
    assert (
        isinstance(field_values, np.ndarray) and field_values.ndim == 2
    ), "Field must be a 2D numpy array"
    assert force_field is None or (
        isinstance(force_field, np.ndarray) and force_field.ndim == 3
    ), "If given, force field must be a 3D numpy array"
    ## Spatial informations
    if bound_limits is not None:
        assert len(bound_limits) == 2, "Bound limits must be a tuple of length 2"
        X, Y = np.meshgrid(
            np.linspace(0, bound_limits[0], field_values.shape[0]),
            np.linspace(0, bound_limits[1], field_values.shape[1]),
            indexing="ij",
        )
    else:
        X, Y = np.meshgrid(
            np.arange(field_values.shape[0]),
            np.arange(field_values.shape[1]),
            indexing="ij",
        )
    ## Plot the concentration map
    fig, ax = plt.subplots(1, 1, dpi=300)
    im = ax.contourf(X, Y, field_values, cmap=cmap, levels=100)
    # Plot the force field if any
    if force_field is not None:
        ax.quiver(
            X,
            Y,
            force_field[0],
            force_field[1],
            angles="xy",
            scale_units="xy",
            scale=1,
        )

    ax.set(title=title, xlabel="x", ylabel="y")
    fig.colorbar(im, ax=ax)
    return fig, ax, im


def plot_walkers(
    positions: np.ndarray,
    concentration: Optional[np.ndarray] = None,
    bound_limits: Optional[Tuple[float]] = None,
    target: Optional[np.ndarray] = None,
    force_field: Optional[np.ndarray] = None,
    cmap_concentration: str = "inferno",
    cmap_walkers: str = "afmhot",
    title: str = "Concentration map + walkers",
):
    """Plot the walkers positions with a color corresponding to their concentration.

    Args:
        positions (np.ndarray): _description_
        concentration (Optional[np.ndarray], optional): _description_. Defaults to None.
        target (Optional[np.ndarray], optional): _description_. Defaults to None.
        force_field (Optional[np.ndarray], optional): _description_. Defaults to None.
        cmap_concentration (str, optional): _description_. Defaults to "inferno".
        cmap_walkers (str, optional): _description_. Defaults to "afmhot".
        title (str, optional): _description_. Defaults to "Concentration map + walkers".

    Returns:
        _type_: _description_
    """
    assert (
        isinstance(positions, np.ndarray) and positions.ndim == 2
    ), "Positions must be a collection of 2D numpy array"
    assert force_field is None or (
        isinstance(force_field, np.ndarray) and force_field.shape == positions.shape
    ), "If given, force field must be a collection of 2D numpy array"
    # Create colormaps
    colormap = plt.get_cmap(cmap_walkers)
    cloud_colors = colormap(np.linspace(0, 1, positions.shape[0]))

    # Plot the concentration map as backgroud if it exists
    if concentration is None:
        fig, ax = plt.subplots(dpi=300)
    else:
        fig, ax, _ = plot_concentration_map(
            concentration,
            bound_limits=bound_limits,
            cmap=cmap_concentration,
            title=title,
        )

    # Plot the force field at walkers positions
    if force_field is not None:
        ax.quiver(
            positions[:, 0],
            positions[:, 1],
            force_field[:, 0],
            force_field[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
        )

    # Print the target if it exists
    if target is not None:
        assert isinstance(target, np.ndarray) and target.shape == (
            2,
        ), "Expected target to be a 2D vector"
        ax.plot(*target, color="green", marker="x", markersize=5, ls="", label="Target")
        fig.legend()

    # Plot the walkers with appropriate colors
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c=cloud_colors, s=10)
    return fig, ax, scatter


def animate_walk_history(
    trajectories: np.ndarray,
    concentration: np.ndarray = None,
    bound_limits: Optional[Tuple[float]] = None,
    output_path: str = None,
    target: Optional[np.ndarray] = None,
    force_field: Optional[np.ndarray] = None,
    cmap_concentration: str = "inferno",
    cmap_walkers: str = "afmhot",
    title: str = "Concentration map + walkers",
    time_step: int = 1,
    fps: int = 30,
):
    assert (
        isinstance(trajectories, np.ndarray)
        and trajectories.ndim == 3
        and trajectories.shape[2] == 2
    ), "Expected trajectories to be a collection of N x T positions 2D array"
    t_max = trajectories.shape[1]

    # Create colormap
    fig, ax, scatter = plot_walkers(
        trajectories[:, 0],
        concentration=concentration,
        bound_limits=bound_limits,
        force_field=force_field,
        target=target,
        cmap_concentration=cmap_concentration,
        cmap_walkers=cmap_walkers,
        title=title,
    )

    def update(frame):
        # read new points
        ax.set_title(f"{title} - t={frame * time_step}")
        scatter.set_offsets(trajectories[:, frame])

    anim = animation.FuncAnimation(fig, update, frames=range(t_max), interval=200)

    if output_path is None:
        return anim
    anim.save(output_path, fps=fps)
    plt.close(fig)


# def plot_3D_trajectory(
#     positions: np.ndarray,
#     goal: np.ndarray,
#     obstacle_mask: np.ndarray,
#     title: str = "3D trajectory",
# ):
#     colormap = plt.get_cmap("viridis")
#     cloud_colors = colormap(np.linspace(0, 1, positions.shape[0]))

#     ax = plt.figure(dpi=300).add_subplot(projection="3d")
#     ax.set(title=title, xlabel="x", ylabel="y", zlabel="z", xlim=(0, 10), ylim=(0, 10))
#     for color, pos in zip(cloud_colors, positions):
#         ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], color=color, s=5)
#     ax.voxels(obstacle_mask, facecolor="red", edgecolor="k")
#     ax.scatter(*goal, s=10, c="green")
#     plt.show()


def animate_volume(
    concentration: np.ndarray,
    axis: int = 2,
    title: str = "Concentration",
):
    assert concentration.ndim == 3, "The concentration must be a 3D array"
    assert axis in [0, 1, 2], "The axis must be 0, 1 or 2"
    # Compute the gradient
    if axis == 0:
        concentration = concentration.transpose((0, 1, 2))
        labels = ["x", "y", "z"]
    elif axis == 1:
        concentration = concentration.transpose((1, 2, 0))
        labels = ["y", "z", "x"]
    else:
        concentration = concentration.transpose((2, 0, 1))
        labels = ["z", "x", "y"]
    # force = np.gradient(concentration)
    # Create colormap
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    z_max = concentration.shape[0]
    vmin, vmax = np.min(concentration), np.max(concentration)
    im = ax.imshow(concentration[0], cmap="inferno", vmax=concentration.max(), vmin=concentration.min())

    def update(frame):
        # read new points
        if frame >= z_max:
            z = 2 * z_max - frame - 1
        else:
            z = frame
        # z = frame % z_max
        # ax.clear()
        ax.set(xlabel=labels[1], ylabel=labels[2], title=f"{title} - {labels[0]}={z}")
        im.set_data(concentration[z])
        
        # ax.imshow(concentration[z], cmap="inferno", vmax=vmax, vmin=vmin)
        # ax.quiver(
        #     force[2][z],
        #     force[1][z],
        #     units="xy",
        #     angles="xy",
        #     scale=1,
        # )

    return animation.FuncAnimation(fig, update, frames=range(2 * z_max), interval=200)

def plot_2D_trajectory(
    positions: np.ndarray,
    goal: np.ndarray,
    obstacles: List[Rectangle],
    concentration: Optional[SampledField] = None,
    title: str = "2D trajectory",
) -> plt.Figure:
    """Plot the trajectory of a walker in 2D with a color corresponding to its concentration."""
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, positions.shape[0]))

    fig, ax = plt.subplots(1, 1, dpi=300)
    if concentration is not None:
        assert isinstance(concentration, SampledField)

        max_bounds = concentration._bounds.max
        im = ax.imshow(
            concentration._values.to_numpy().T,
            cmap="inferno",
            extent=[0, max_bounds[0], max_bounds[1], 0],
        )
        plt.colorbar(im, ax=ax, label="Concentration (log)")

    for i in range(positions.shape[0]):
        ax.plot(positions[i, :, 0], positions[i, :, 1], color=colors[i], marker=".")

        # ax.scatter(
        #     positions[i, :, 0],
        #     positions[i, :, 1],
        #     s=5,
        #     color=colors[i],
        # )
    ax.add_patch(plt.Circle(goal, 0.5, color="green"))
    for obs in obstacles:
        ax.add_patch(plt.Rectangle((obs.x, obs.y), obs.width, obs.height, color="red"))
    ax.set(title=title, xlabel="x", ylabel="y")
    # fig.show()

    return fig


def plot_3D_trajectory(
    trajectories: np.ndarray,
    goal: np.ndarray,
    obstacles: List[Cube],
    title: str = "3D trajectory",
) -> go.Figure:
    fig = go.Figure()
    for i in range(trajectories.shape[0]):
        fig.add_trace(
            go.Scatter3d(
                x=trajectories[i, :, 0],
                y=trajectories[i, :, 1],
                z=trajectories[i, :, 2],
                mode="lines",
                name=f"walker {i}",
                # colorscale="Viridis",
                # color=color,
            )
        )
    goal = goal[None]
    fig.add_trace(
        go.Scatter3d(
            x=goal[:, 0],
            y=goal[:, 1],
            z=goal[:, 2],
            mode="markers",
            marker=dict(size=10, color="green"),
            name="goal",
        )
    )
    for obs in obstacles:
        fig.add_trace(
            go.Mesh3d(
                x=obs.vertices[:, 0],
                y=obs.vertices[:, 1],
                z=obs.vertices[:, 2],
                i=obs.indices[:, 0],
                j=obs.indices[:, 1],
                k=obs.indices[:, 2],
                opacity=0.5,
                color="red",
                name="obstacle",
            )
        )
    # fig.show()

    return fig


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
    from pathlib import Path

    dirpath = Path("/home/rcremese/projects/snake-ai/simulations").resolve(strict=True)
    filepath = dirpath.joinpath(
        "Slot(20,20)_pixel_Tmax=800.0_D=1", "seed_10", "field.npz"
    )
    obj = np.load(filepath)

    field = obj["data"]
    force_field = np.stack(np.gradient(field))
    fig, _, _ = plot_concentration_map(field, obj["upper"], force_field)

    positions = np.array([(x, y) for x in range(20) for y in range(20)])
    sampled_field = np.array([force_field[:, x * 10, y * 10] for x, y in positions])
    plot_walkers(
        positions,
        field,
        obj["upper"],
        # force_field=sampled_field,
        target=np.array([10, 10]),
    )
    plt.show()
