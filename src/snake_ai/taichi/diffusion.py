import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from typing import Tuple, List
from matplotlib import animation


def get_obstacle_free_laplacian_matrix(*resolution: Tuple[int]) -> sp.lil_matrix:
    assert len(resolution) in [2, 3], "The resolution must be a tuple of length 2 or 3"
    assert all(
        isinstance(r, int) and r > 0 for r in resolution
    ), "The resolution must be a tuple of positive integers"
    ex = np.ones(resolution[0])
    ey = np.ones(resolution[1])
    Dxx = sp.diags([ex, -2 * ex, ex], [-1, 0, 1], shape=(resolution[0], resolution[0]))
    Dyy = sp.diags([ey, -2 * ey, ey], [-1, 0, 1], shape=(resolution[1], resolution[1]))
    laplace_2d = sp.kronsum(Dyy, Dxx, format="lil")
    if len(resolution) == 2:
        return laplace_2d

    ez = np.ones(resolution[2])
    Dzz = sp.diags([ez, -2 * ez, ez], [-1, 0, 1], shape=(resolution[2], resolution[2]))
    return sp.kronsum(Dzz, laplace_2d, format="lil")


def get_absorbing_obstacles_laplacian_matrix(binary_map: np.ndarray) -> sp.lil_matrix:
    """Compute the laplacian matrix associated with an environment with obstacles and absobing boundary conditions

    Args:
        binary_map (np.ndarray): 2d or 3D array of shape with 1 for obstacles and 0 for free space. Set the resolution of the environment.

    Returns:
        sp.lil_matrix: NxN laplacian matrix where N is the number of cells in the environment
    """
    assert binary_map.ndim in [2, 3], "The binary map must be 2D or 3D"
    resolution = binary_map.shape

    def coord2ind(coord: Tuple[int]):
        assert len(coord) == len(
            resolution
        ), "The coordinate must have the same dimension as the resolution"
        if len(resolution) == 2:
            i, j = coord
            return i * resolution[1] + j
        elif len(resolution) == 3:
            i, j, k = coord
            return i * resolution[1] * resolution[2] + j * resolution[2] + k
        else:
            raise ValueError("The resolution must be 2D or 3D")

    laplace = get_obstacle_free_laplacian_matrix(*resolution)
    # Remove the positions that are on the obstacles
    positions = np.argwhere(binary_map)

    for pos in positions:
        ind = coord2ind(pos)
        laplace[:, ind] = 0
        laplace[ind, :] = 0
        laplace[ind, ind] = -2 * binary_map.ndim
    return laplace


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
    force = np.gradient(concentration)
    # Create colormap
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    z_max = concentration.shape[0]
    vmin, vmax = np.min(concentration), np.max(concentration)

    def update(frame):
        # read new points
        if frame >= z_max:
            z = 2 * z_max - frame - 1
        else:
            z = frame
        # z = frame % z_max
        ax.clear()
        ax.set(xlabel=labels[1], ylabel=labels[2], title=f"{title} - {labels[0]}={z}")
        ax.imshow(concentration[z], cmap="inferno", vmax=vmax, vmin=vmin)
        ax.quiver(
            force[2][z],
            force[1][z],
            units="xy",
            angles="xy",
            scale=1,
        )

    anim = animation.FuncAnimation(fig, update, frames=range(2 * z_max), interval=200)
    return anim


def main():
    row, col, depth = 10, 20, 40

    def coord2index(i, j, k):
        return i * col * depth + j * depth + k

    source = np.zeros((row, col, depth), dtype=np.float32)
    source[5, 10, 15] = 100

    obstacles = np.zeros((row, col, depth), dtype=np.float32)
    obstacles[4:6, 4:6, 4:6] = 1
    obstacles[0:2, 10:12, 10:12] = 1
    obstacles[7:, 15:17, 15:17] = 1

    laplace = get_absorbing_obstacles_laplacian_matrix(obstacles)

    solver = sp.linalg.factorized(-laplace.tocsc())
    solution = solver(source.flatten())
    solution = solution.reshape(row, col, depth)
    # Five stencil laplace filter
    laplace_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # Nine stencil laplace filter
    # laplace_filter = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    # smoothed_sol = sg.convolve2d(solution, -laplace_filter, mode="same")
    smoothed_sol = np.log(np.where(solution > 1e-6, solution, 1e-6))
    # gradx, grady = np.gradient(smoothed_sol)

    anim = animate_volume(smoothed_sol, axis=2)
    # anim = animate_volume(solution, axis=2)

    # fig, ax = plt.subplots(1, 4, figsize=(12, 4))

    # ax[0].imshow(source, cmap="inferno")
    # ax[0].set(title="source")
    # ax[1].imshow(laplace.toarray(), cmap="inferno")
    # ax[1].set(title="laplace matrix")
    # ax[2].imshow(solution, cmap="inferno")
    # ax[2].set(title="solution")
    # ax[3].imshow(smoothed_sol, cmap="inferno")
    # ax[3].quiver(grady, gradx, units="xy", angles="xy", scale=1)
    # ax[3].set(title="Solution in log scale")
    plt.show()


if __name__ == "__main__":
    main()
