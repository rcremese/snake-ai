from snake_ai.envs import GridWorld3D, GridWorld
from snake_ai.envs.converter import EnvConverter
from snake_ai.taichi.field import ScalarField

import scipy.sparse as sp
import scipy.sparse.linalg as spl
import numpy as np

from typing import Optional, Tuple, List, Union
import logging, copy

import matplotlib.pyplot as plt
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


def get_laplacian_matrix_from_obstacle_binary_map(
    binary_map: np.ndarray,
) -> sp.lil_matrix:
    """Compute the laplacian matrix associated with an environment with obstacles and absobing boundary conditions

    Args:
        binary_map (np.ndarray): 2D or 3D array of shape with 1 for obstacles and 0 for free space. Set the resolution of the environment.

    Returns:
        sp.lil_matrix: NxN laplacian matrix where N is the number of cells in the environment
    """
    assert binary_map.ndim in [2, 3], "The binary map must be 2D or 3D"
    resolution = binary_map.shape

    laplace = get_obstacle_free_laplacian_matrix(*resolution)
    # Remove the positions that are on the obstacles
    positions = np.argwhere(binary_map)
    indices = np.ravel_multi_index(positions.T, resolution)

    for ind in indices:
        # ind = coord2ind(pos)
        laplace[:, ind] = 0
        laplace[ind, :] = 0
        laplace[ind, ind] = -2 * binary_map.ndim
    return laplace


def get_laplacian_matrix_from_obstacle_positions(
    positions: np.ndarray, resolution: Tuple[int]
) -> sp.lil_matrix:
    """Compute the laplacian matrix associated with an environment with obstacles and absorbing boundary conditions from positions of positive values

    Args:
        positions (np.ndarray): positions in the environment of obstacles at a given resolution.
        2D array of shape (N, 2) or (N, 3) with N the number of points in obstacles
        resolution (Tuple[int]): shape of the matrix

    Returns:
        sp.lil_matrix: Matrix associated with the environment
    """
    assert positions.ndim == 2, "The positions must be a 2D array"
    assert positions.shape[1] == len(
        resolution
    ), "The positions must have the same dimension as the resolution"

    laplace = get_obstacle_free_laplacian_matrix(*resolution)
    indices = np.ravel_multi_index(positions.T, resolution)
    for ind in indices:
        laplace[:, ind] = 0
        laplace[ind, :] = 0
        laplace[ind, ind] = -2 * positions.shape[1]
    return laplace


class DiffusionSolver:
    def __init__(
        self,
        env: Union[GridWorld, GridWorld3D],
        resolution: Optional[Union[int, Tuple[int]]] = None,
    ) -> None:
        assert isinstance(
            env, (GridWorld, GridWorld3D)
        ), "The environment must be a GridWorld or GridWorld3D instance"
        self.converter = EnvConverter(env, resolution)
        # if isinstance(env, GridWorld):
        #     self.converter = Env2DConverter(env, resolution)
        # elif isinstance(env, GridWorld3D):
        #     self.converter = Env3DConverter(env, resolution)
        self._obstacles = copy.deepcopy(env.obstacles)
        self._update_solver()

    def solve(self, shape="box", init_value: float = 1.0) -> np.ndarray:
        if shape.lower() not in [
            "box",
            "point",
        ]:
            raise ValueError("Shape must be either 'box' or 'point'.")
        assert init_value > 0, "The initial value must be positive"

        if self.converter.env.obstacles != self._obstacles:
            self._update_solver()
            self._obstacles = copy.deepcopy(self.converter.env.obstacles)
        # goal = self.converter.convert_goal_to_binary_map(shape)

        source = self.converter.convert_goal_to_binary_map(shape)
        values = self._solver(source.flatten())
        return ScalarField(values.reshape(self.resolution), self.converter.env.bounds)

    ## Properties
    @property
    def resolution(self) -> Tuple[int]:
        return self.converter.resolution

    @resolution.setter
    def resolution(self, resolution: Union[int, Tuple[int]]):
        try:
            self.converter.resolution = resolution
        except (ValueError, TypeError) as exception:
            raise ValueError(exception) from exception
        self._update_solver()

    ## Private methods
    def _update_solver(self):
        logging.debug("Updating the solver")
        binary_map = self.converter.convert_obstacles_to_binary_map()
        # positions = self.converter.convert_obstacles_to_indices()
        # laplace = get_absorbing_obstacles_laplacian_matrix_from_positions(
        #     positions, self.resolution
        # )
        laplace = get_laplacian_matrix_from_obstacle_binary_map(binary_map)
        self._solver = spl.factorized(-laplace.tocsc())


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


def plot_volume(concentration: np.ndarray):
    fig, ax = plt.subplots(1, 3, figsize=(8, 8))
    ind = np.argmax(concentration)
    i, j, k = np.unravel_index(ind, concentration.shape)

    ax[0].imshow(concentration[i], cmap="inferno")
    ax[0].scatter(k, j, color="red", marker="x")
    ax[0].set(title=f"max x = {i}", xlabel="z", ylabel="y")
    ax[1].imshow(concentration[:, j, :], cmap="inferno")
    ax[1].scatter(k, i, color="red", marker="x")
    ax[1].set(title=f"max y = {j}", xlabel="z", ylabel="x")
    ax[2].imshow(concentration[:, :, k], cmap="inferno")
    ax[2].scatter(j, i, color="red", marker="x")
    ax[2].set(title=f"max z = {k}", xlabel="y", ylabel="x")
    plt.show()


def main():
    from snake_ai.envs.random_obstacles_3d import RandomObstacles3D
    import time

    row, col, depth = 10, 10, 10
    env = RandomObstacles3D(row, col, depth, nb_obs=10, max_size=2)
    env.reset()
    converter = EnvConverter(env, resolution=40)

    plt.show()

    source = converter.convert_goal_to_binary_map("point")

    tic = time.perf_counter()
    binary_map = converter.convert_obstacles_to_binary_map()
    laplace = get_laplacian_matrix_from_obstacle_binary_map(binary_map)
    toc = time.perf_counter()
    print(
        f"Time to compute the {laplace.shape} laplacian matrix with dense matrix: {toc - tic:0.5f} seconds"
    )

    tic = time.perf_counter()
    positions = converter.convert_obstacles_to_indices()
    laplace = get_laplacian_matrix_from_obstacle_positions(
        positions, converter.resolution
    )
    toc = time.perf_counter()
    print(
        f"Time to compute the {laplace.shape} laplacian matrix with position vector : {toc - tic:0.5f} seconds"
    )

    tic = time.perf_counter()
    solver = spl.factorized(-laplace.tocsc())
    toc = time.perf_counter()
    print(f"Time to factorize the laplacian matrix: {toc - tic:0.5f} seconds")

    tic = time.perf_counter()
    solution = solver(source.flatten())
    toc = time.perf_counter()
    print(f"Time to solve the equation with factorisation : {toc - tic:0.5f} seconds")

    tic = time.perf_counter()
    solution = spl.spsolve(-laplace, source.flatten())
    toc = time.perf_counter()
    print(
        f"Time to solve the equation without factorisation : {toc - tic:0.5f} seconds"
    )
    solution = solution.reshape(converter.resolution)
    # solution = solver.solve()

    # source = np.zeros((row, col, depth), dtype=np.float32)
    # source[5, 10, 15] = 100

    # obstacles = np.zeros((row, col, depth), dtype=np.float32)
    # obstacles[4:6, 4:6, 4:6] = 1
    # obstacles[0:2, 10:12, 10:12] = 1
    # obstacles[7:, 15:17, 15:17] = 1

    # laplace = get_absorbing_obstacles_laplacian_matrix(obstacles)

    # solver = sp.linalg.factorized(-laplace.tocsc())
    # solution = solver(source.flatten())
    # solution = solution.reshape(row, col, depth)
    # Five stencil laplace filter
    laplace_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # Nine stencil laplace filter
    # laplace_filter = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    # smoothed_sol = sg.convolve2d(solution, -laplace_filter, mode="same")
    smoothed_sol = np.log(np.where(solution > 1e-6, solution, 1e-6))

    plot_volume(smoothed_sol)
    anim = animate_volume(smoothed_sol, axis=2)

    plt.show()


if __name__ == "__main__":
    main()
