import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from typing import List, Tuple, Optional


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
