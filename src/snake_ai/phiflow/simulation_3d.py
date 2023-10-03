from snake_ai.envs.grid_world_3d import GridWorld3D
from phi import flow
from typing import Optional, Union, Tuple, List
import scipy.sparse as sp
import numpy as np


class Env3DConverter:
    def __init__(self, env: GridWorld3D) -> None:
        assert isinstance(
            env, GridWorld3D
        ), f"Environmnent must be of type GridWorld3D, not {type(env)}"
        self.env = env

    def convert_obstacles_to_binary_map(
        self, resolution: Optional[Union[int, Tuple[int]]] = None
    ) -> flow.field.SampledField:
        res = self._check_resolution(resolution)

        grid = flow.CenteredGrid(
            0,
            bounds=flow.Box(x=self.env.width, y=self.env.height, z=self.env.depth),
            x=res[0],
            y=res[1],
            z=res[2],
        )
        if self.env.nb_obstacles == 0:
            return grid

        obstacles = self.convert_obstacles_to_geometries()
        return flow.field.resample(flow.union(obstacles), grid)

    def convert_source_to_binary_map(
        self, resolution: Optional[Union[int, Tuple[int]]] = None, shape: str = "sphere"
    ) -> flow.field.SampledField:
        assert shape.lower() in [
            "sphere",
            "box",
        ], f"Shape must be either 'sphere' or 'box', not {shape}"

        res = self._check_resolution(resolution)

        grid = flow.CenteredGrid(
            0,
            bounds=flow.Box(x=self.env.width, y=self.env.height, z=self.env.depth),
            x=res[0],
            y=res[1],
            z=res[2],
        )
        if shape.lower() == "sphere":
            source = self.convert_source_to_sphere()
        else:
            source = self.convert_source_to_box()
        return flow.field.resample(source, grid)

    def convert_obstacles_to_geometries(self) -> List[flow.Box]:
        return [
            flow.Box(
                x=(obs.x, obs.x + obs.width),
                y=(obs.y, obs.y + obs.height),
                z=(obs.z, obs.z + obs.depth),
            )
            for obs in self.env.obstacles
        ]

    def convert_source_to_box(self) -> flow.Box:
        return flow.Box(
            x=(self.env.goal.x, self.env.goal.x + self.env.goal.width),
            y=(self.env.goal.y, self.env.goal.y + self.env.goal.height),
            z=(self.env.goal.z, self.env.goal.z + self.env.goal.depth),
        )

    def convert_source_to_sphere(self) -> flow.Sphere:
        center = self.env.center
        min_dist = min(self.env.width, self.env.height, self.env.depth)

        return flow.Sphere(
            x=center[0],
            y=center[1],
            z=center[2],
            radius=min_dist / 2,
        )

    ## Private mlethods
    def _check_resolution(self, resolution: Union[int, Tuple[int]]) -> Tuple[int]:
        if resolution is None:
            return (self.env.height, self.env.width, self.env.depth)
        elif isinstance(resolution, int):
            assert resolution > 0, "Resolution must be a positive integer"
            return (resolution, resolution, resolution)
        elif isinstance(resolution, tuple):
            assert len(resolution) == 3, "Resolution must be a tuple of length 3"
            assert all(
                isinstance(res, int) and res > 0 for res in resolution
            ), f"Resolution must be a tuple of positive integers. Get {resolution}"
            return resolution
        else:
            raise TypeError(
                "Resolution must be either an integer or a tuple of integers"
            )


def solve_diffusion(
    source: flow.CenteredGrid, obstacle_mask: flow.CenteredGrid, max_iter: int = 1000
) -> flow.CenteredGrid:
    @flow.math.jit_compile_linear
    def forward(concentration: flow.CenteredGrid, obstacle_mask: flow.CenteredGrid):
        return flow.field.where(
            obstacle_mask, concentration, -flow.field.laplace(concentration)
        )

    return flow.math.solve_linear(
        forward,
        source,
        solve=flow.math.Solve(x0=source, max_iterations=max_iter),
        obstacle_mask=obstacle_mask,
    )


class DiffusionSolver:
    def __init__(self, env: GridWorld3D) -> None:
        assert isinstance(
            env, GridWorld3D
        ), f"Environmnent must be of type GridWorld3D, not {type(env)}"
        self.env = env

    def solve(self, resolution: Tuple[int] = None) -> np.ndarray:
        converter = Env3DConverter(self.env)

        obstacle_mask = converter.convert_obstacles_to_binary_map(resolution)
        source = converter.convert_source_to_binary_map(resolution, shape="sphere")

        laplace = DiffusionSolver.get_laplace_matrix(
            obstacle_mask.values.numpy("x,y,z")
        )
        values = source.values.numpy("x,y,z")
        shape = values.shape

        solution = sp.linalg.spsolve(-laplace, values.flatten())
        return source.with_values(
            flow.tensor(solution.reshape(shape), flow.spatial("x,y,z"))
        )
        # return solution.reshape(shape)

    @staticmethod
    def get_laplace_matrix(obstacle_mask: np.ndarray) -> sp.csc_matrix:
        assert (
            obstacle_mask.ndim == 3
        ), f"Obstacle mask must be a 3D array, not {obstacle_mask.ndim}D"
        resolution = obstacle_mask.shape
        N = np.multiply.reduce(resolution)

        def pos_to_index(i, j, k) -> int:
            return i + j * resolution[0] + k * resolution[0] * resolution[1]

        ones = np.ones(N)
        diags = [ones, ones, ones, -6 * ones, ones, ones, ones]
        offsets = [
            -resolution[0] * resolution[1],
            -resolution[0],
            -1,
            0,
            1,
            resolution[0],
            resolution[0] * resolution[1],
        ]
        laplace = sp.dia_matrix((diags, offsets), shape=(N, N))
        laplace: sp.lil_matrix = laplace.tolil()

        ## Remove the positions that are on the edge of the grid
        for index in range(resolution[0], N, resolution[0]):
            laplace[index - 1, index] = 0
            laplace[index, index - 1] = 0
            if index % resolution[1] == 0:
                laplace[index - resolution[0], index] = 0
                laplace[index, index - resolution[0]] = 0
        ## Set the laplace matrix to identity for the obstacle positions
        positions = np.argwhere(obstacle_mask)
        for i, j, k in positions:
            ind = pos_to_index(i, j, k)

            laplace[:, ind] = 0
            laplace[ind, :] = 0
            laplace[ind, ind] = -6
        return laplace.tocsc()


if __name__ == "__main__":
    from snake_ai.envs.random_obstacles_3d import RandomObstacles3D
    import matplotlib.pyplot as plt

    env = RandomObstacles3D(10, 10, 10, nb_obs=10, max_size=2)

    converter = Env3DConverter(env)
    env.reset()
    binary_map = converter.convert_obstacles_to_binary_map(10)
    source = converter.convert_source_to_binary_map(10, shape="sphere")

    # laplace = DiffusionSolver.get_laplace_matrix(binary_map.values.numpy("x,y,z"))
    solver = DiffusionSolver(env)
    # concentration = solver.solve()
    concentration = solve_diffusion(10 * source, binary_map)
    force_field = flow.field.spatial_gradient(concentration)
    # vmin, vmax = concentration.min(), concentration.max()
    print(env.obstacles)
    print(concentration.shape)
    # fig1, ax1 = plt.subplots(2, 5)
    # for i in range(10):
    #     im = ax1[i // 5, i % 5].imshow(
    #         concentration[:, :, i], vmin=vmin, vmax=vmax, cmap="inferno"
    #     )
    #     ax1[i // 5, i % 5].set(title=f"z = {i}")
    # fig1.colorbar(im, ax=ax1.ravel().tolist())

    # obstacles = binary_map.values.numpy("x,y,z")
    # fig2, ax2 = plt.subplots(2, 5)
    # for i in range(10):
    #     im = ax2[i // 5, i % 5].imshow(obstacles[i, :, :])
    #     ax2[i // 5, i % 5].set(title=f"z = {i}")

    # concentration = solve_diffusion(env)
    flow.vis.plot(binary_map, concentration, force_field)

    plt.show()
