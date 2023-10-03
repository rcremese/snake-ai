from snake_ai.envs import GridWorld, Rectangle
from snake_ai.envs.geometry import Cube
from snake_ai.envs.grid_world_3d import GridWorld3D
import numpy as np

from typing import Union, Tuple, List, Optional


def convert_free_space_to_point_cloud(env: GridWorld, step: int = 1) -> np.ndarray:
    """Transform free space of the environment into a point cloud.

    Args:
        env (GridWorld): Environment in 2D containing free positons.
    Returns:
        np.ndarray: _description_
    """
    assert isinstance(env, GridWorld), "environmnent must be of type GridWorld"
    assert env.free_positions is not None, "Environment does not contain free positions"
    assert isinstance(step, int) and step > 1, "Step must be an integer > 1"
    positions = []
    for x, y in env.free_positions:
        if x % step == 0 and y % step == 0:
            positions.append((x + 0.5, y + 0.5))
    return np.array(positions)


def convert_obstacles_to_physical_space(env: GridWorld) -> List[Rectangle]:
    assert isinstance(env, GridWorld), "environmnent must be of type GridWorld"
    assert (
        env.obstacles is not None
    ), "Environment does not contain obstacles. Call reset() first."
    return [
        Rectangle(
            obstacle.x // env.pixel,
            obstacle.y // env.pixel,
            obstacle.width // env.pixel,
            obstacle.height // env.pixel,
        )
        for obstacle in env.obstacles
    ]


def convert_obstacles_to_binary_map(env: GridWorld, res: str = "pixel") -> np.ndarray:
    assert isinstance(env, GridWorld), "Environmnent must be of type GridWorld"
    assert env.obstacles is not None, "Environment does not contain obstacles"
    assert res.lower() in [
        "pixel",
        "meta",
    ], "Resolution must be either 'pixel' or 'meta'"
    resolution = 1 if res.lower() == "pixel" else env.pixel
    binary_map = np.zeros((env.height * resolution, env.width * resolution))
    for obstacle in env.obstacles:
        x = int(obstacle.x / env.pixel) * resolution
        y = int(obstacle.y / env.pixel) * resolution
        width = int(obstacle.width / env.pixel) * resolution
        height = int(obstacle.height / env.pixel) * resolution
        binary_map[y : y + height, x : x + width] = 1
    return binary_map


def convert_agent_position(env: GridWorld) -> np.ndarray:
    assert isinstance(env, GridWorld), "environmnent must be of type GridWorld"
    assert (
        env.agent is not None
    ), "The agent is not initialized in the environment. Call reset first."
    return np.array(env.agent.position.center) / np.array([env.pixel, env.pixel])


def convert_goal_position(env: GridWorld) -> np.ndarray:
    assert isinstance(env, GridWorld), "environmnent must be of type GridWorld"
    assert (
        env.goal is not None
    ), "The goal is not initialized in the environment. Call reset first."
    return np.array(env.goal.center) / np.array([env.pixel, env.pixel])


class GridWorld3DConverter:
    def __init__(self, env: GridWorld3D) -> None:
        assert isinstance(
            env, GridWorld3D
        ), f"Environmnent must be of type GridWorld3D, not {type(env)}"
        self.env = env

    ## Public methods
    def convert_3d_obstacles_to_binary_map(
        self, resolution: Optional[Union[int, Tuple[int]]] = None
    ) -> np.ndarray:
        assert (
            self.env._obstacles is not None
        ), "Environment does not contain obstacles. Reset environment first."
        resolution = self._check_resolution(resolution)

        steps = self.convert_resolution_to_step(resolution)
        # resolution = self._check_resolution(resolution)

        binary_map = np.zeros(resolution, dtype=int)
        if self.env.nb_obstacles == 0:
            return binary_map

        for obstacle in self.env.obstacles:
            min_ind = np.floor(obstacle.min / steps).astype(int)
            max_ind = np.ceil(obstacle.max / steps).astype(int)
            binary_map[
                min_ind[0] : max_ind[0],
                min_ind[1] : max_ind[1],
                min_ind[2] : max_ind[2],
            ] = 1
        return binary_map

    def convert_resolution_to_step(self, resolution: Tuple[int]) -> np.ndarray:
        assert (
            len(resolution) == 3
        ), f"Resolution must be a tuple of positive integers. Get {resolution}"
        res = np.array(resolution, dtype=float)
        assert np.all(res > 0), f"Resolutions must be a tuple of positive integers"
        return (self.env.bounds.max - self.env.bounds.min) / (res - 1)

    def _check_resolution(self, resolution: Union[int, Tuple[int]]) -> Tuple[int]:
        """Convert an input resolution to a tuple of 3 integers

        Args:
            resolution (Union[int, Tuple[int]]): desired resolution for a force field

        Raises:
            TypeError: if the resolution is not None, an integer or a tuple of integers

        Returns:
            Tuple[int]: resolution e
        """
        if resolution is None:
            res = (self.env.height, self.env.width, self.env.depth)
        elif isinstance(resolution, int):
            assert resolution > 0, "Resolution must be a positive integer"
            res = (resolution, resolution, resolution)
        elif isinstance(resolution, tuple):
            assert len(resolution) == 3, "Resolution must be a tuple of length 3"
            assert all(
                isinstance(res, int) and res > 0 for res in resolution
            ), f"Resolution must be a tuple of positive integers. Get {resolution}"
            res = resolution
        else:
            raise TypeError(
                "Resolution must be either an integer or a tuple of integers"
            )
        return res


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from snake_ai.envs.random_obstacles_3d import RandomObstacles3D

    N = 10

    env = RandomObstacles3D(10, 10, 10, nb_obs=5, max_size=1)
    env.reset()
    converter = GridWorld3DConverter(env)
    binary_map = converter.convert_3d_obstacles_to_binary_map(N**2)

    fig, ax = plt.subplots(N, N)
    for i in range(N):
        for j in range(N):
            z = i + N * j
            ax[i, j].imshow(binary_map[:, :, z])
            ax[i, j].set(title=f"z= {z}", xlabel="y", ylabel="x")
    print(env.obstacles)
    plt.show()
