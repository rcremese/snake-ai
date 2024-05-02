from snake_ai.envs import GridWorld, Rectangle
from snake_ai.envs.geometry import Cube
from snake_ai.envs.grid_world_3d import GridWorld3D

import numpy as np
from abc import abstractmethod, ABC
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


class EnvConverter:
    def __init__(
        self,
        env: Union[GridWorld, GridWorld3D],
        resolution: Optional[Union[int, Tuple[int]]] = None,
    ) -> None:
        assert isinstance(
            env, (GridWorld, GridWorld3D)
        ), f"Environmnent must be of type GridWorld or GridWorld3D, not {type(env)}"
        self.env = env
        self.dim = 2 if isinstance(env, GridWorld) else 3
        # Set the resolution of the converter
        if resolution is None:
            if self.dim == 2:
                resolution = (env.height, env.width)
            else:
                resolution = (env.height, env.width, env.depth)
        self.resolution = resolution

    ## Public methods
    def convert_obstacles_to_binary_map(self) -> np.ndarray:
        assert (
            self.env.obstacles is not None
        ), "Environment does not contain obstacles. Reset environment first."

        binary_map = np.zeros(self.resolution, dtype=bool)
        if self.env.nb_obstacles == 0:
            return binary_map
        indices = self.convert_obstacles_to_indices()
        for ind in indices:
            if (
                ind[0] < 0
                or ind[1] < 0
                or ind[0] >= self.resolution[0]
                or ind[1] >= self.resolution[1]
            ):
                continue
            if self.dim == 2:
                binary_map[ind[0], ind[1]] = 1
            elif self.dim == 3:
                if ind[2] < 0 or ind[2] >= self.resolution[2]:
                    continue
                binary_map[ind[0], ind[1], ind[2]] = 1
            else:
                raise ValueError("Dimension must be either 2 or 3")
        return binary_map

    def convert_obstacles_to_indices(self) -> np.ndarray:
        indices = []
        pixel = self.env.pixel if self.dim == 2 else 1
        for obstacle in self.env.obstacles:
            pixel_min = np.floor(obstacle.min / (self.steps * pixel)).astype(int)
            pixel_max = np.ceil(obstacle.max / (self.steps * pixel)).astype(int)
            if self.dim == 2:
                new_indices = [
                    (i, j)
                    for i in range(pixel_min[0], pixel_max[0])
                    for j in range(pixel_min[1], pixel_max[1])
                ]
            elif self.dim == 3:
                new_indices = [
                    (i, j, k)
                    for i in range(pixel_min[0], pixel_max[0])
                    for j in range(pixel_min[1], pixel_max[1])
                    for k in range(pixel_min[2], pixel_max[2])
                ]
            else:
                raise ValueError("Dimension must be either 2 or 3")
            indices.extend(new_indices)
        return np.array(indices)

    def convert_goal_to_binary_map(self, shape: str) -> np.ndarray:
        assert (
            self.env.goal is not None
        ), "Environment does not contain obstacles. Reset environment first."
        assert shape.lower() in [
            "box",
            "point",
        ], "Shape must be either 'box' or 'point'"

        binary_map = np.zeros(self.resolution, dtype=bool)
        indices = self.convert_goal_to_indices(shape)
        if self.dim == 2:
            binary_map[indices[:, 0], indices[:, 1]] = 1
        elif self.dim == 3:
            binary_map[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        else:
            raise ValueError("Dimension must be either 2 or 3")
        return binary_map

    def convert_goal_to_indices(self, shape: str) -> np.ndarray:
        assert (
            self.env.goal is not None
        ), "Environment does not contain obstacles. Reset environment first."
        assert shape.lower() in [
            "box",
            "point",
        ], "Shape must be either 'box' or 'point'"
        pixel = self.env.pixel if self.dim == 2 else 1
        ## Case in which all the pixel of the goal is represented
        if shape.lower() == "box":
            pixel_min = np.floor(self.env.goal.min / (self.steps * pixel)).astype(int)
            pixel_max = np.ceil(self.env.goal.max / (self.steps * pixel)).astype(int)
            if self.dim == 2:
                indices = [
                    (i, j)
                    for i in range(pixel_min[0], pixel_max[0])
                    for j in range(pixel_min[1], pixel_max[1])
                ]
            elif self.dim == 3:
                indices = [
                    (i, j, k)
                    for i in range(pixel_min[0], pixel_max[0])
                    for j in range(pixel_min[1], pixel_max[1])
                    for k in range(pixel_min[2], pixel_max[2])
                ]
            else:
                raise ValueError("Dimension must be either 2 or 3")
            return np.array(indices)
        ## Case in which only the center of the goal pixel is considered
        elif shape.lower() == "point":
            pixel = np.floor(self.env.goal.center / (self.steps * pixel)).astype(int)
            return np.array([pixel])
        else:
            raise ValueError("Shape must be either 'box' or 'point'")

    def convert_free_positions_to_point_cloud(self, step: int = 1) -> np.ndarray:
        """Transform free space of the environment into a point cloud.

        Args:
            step (int): Step between 2 positions.
        Returns:
            np.ndarray: free positions as a point cloud [N, 2]
        """
        assert isinstance(step, int) and step > 0, "Step must be an integer > 0"
        positions = []
        # if self.dim == 2:
        #     for x, y in self.env.free_positions:
        #         if x % step == 0 and y % step == 0:
        #             positions.append((x + 0.5, y + 0.5))
        # elif self.dim == 3:
        #     for x, y, z in self.env.free_positions:
        #         if x % step == 0 and y % step == 0 and z % step == 0:
        #             positions.append((x + 0.5, y + 0.5, z + 0.5))
        # else:
        #     raise ValueError("Dimension must be either 2 or 3")
        # return np.array(positions)
        for indices in self.env.free_positions:
            if all(ind % step == 0 for ind in indices):
                positions.append(indices)
        return np.array(positions) + 0.5

    def get_agent_position(self, repeats: int = 1) -> np.ndarray:
        if self.dim == 2:
            center = np.array(self.env.agent.position.center) / self.env.pixel
        else:
            center = self.env.agent.center
        return np.repeat(center[None], axis=0, repeats=repeats)

    def get_goal_position(self) -> np.ndarray:
        pixel = self.env.pixel if self.dim == 2 else 1
        return self.env.goal.center / pixel

    # @abstractmethod
    # def convert_obstacles_to_binary_map(self) -> np.ndarray:
    #     raise NotImplementedError

    # @abstractmethod
    # def convert_goal_to_binary_map(self) -> np.ndarray:
    #     raise NotImplementedError

    # @abstractmethod
    # def convert_free_positions_to_point_cloud(self, step: int = 1) -> np.ndarray:
    #     """Transform free space of the environment into a point cloud.

    #     Args:
    #         step (int): Step between 2 positions.
    #     Returns:
    #         np.ndarray: free positions as a point cloud [N, 2 or 3]
    #     """
    #     raise NotImplementedError

    # @abstractmethod
    # def get_agent_position(self, repeats: int = 1) -> np.ndarray:
    #     raise NotImplementedError

    # @abstractmethod
    # def get_goal_position(self) -> np.ndarray:
    #     raise NotImplementedError

    ## Properties
    @property
    def resolution(self) -> Tuple[int]:
        return self._resolution

    @resolution.setter
    def resolution(self, resolution: Union[int, Tuple[int]]):
        if isinstance(resolution, int):
            if resolution <= 0:
                raise ValueError("Resolution must be a positive integer")
            self._resolution = tuple(
                [resolution * dim for dim in self.env.bounds.dimension]
            )
        elif isinstance(resolution, tuple):
            if len(resolution) != self.dim:
                raise ValueError(f"Resolution must be a tuple of length {self.dim}")
            if not all(isinstance(res, int) and res > 0 for res in resolution):
                raise ValueError(
                    f"Resolution must be a tuple of positive integers. Get {resolution}"
                )
            self._resolution = resolution
        else:
            raise TypeError(
                "Resolution must be either an integer or a tuple of integers"
            )

    @property
    def linspace(self) -> List[np.ndarray]:
        """Linspace of the environment as returned by np.linspace"""
        linspaces = []
        for i in range(self.dim):
            linspaces.append(
                np.linspace(
                    self.env.bounds.min[i],
                    self.env.bounds.max[i],
                    self.resolution[i],
                    endpoint=True,
                )
            )
        return linspaces

    @property
    def meshgrid(self) -> List[np.ndarray]:
        """Meshgrid of the environment as returned by np.meshgrid"""
        return np.meshgrid(*self.linspace, indexing="ij")

    @property
    def steps(self) -> np.ndarray:
        return (self.env.bounds.max - self.env.bounds.min) / np.array(self.resolution)


class Env2DConverter(EnvConverter):
    ## Public methods
    def convert_obstacles_to_binary_map(self) -> np.ndarray:
        assert (
            self.env.obstacles is not None
        ), "Environment does not contain obstacles. Reset environment first."

        binary_map = np.zeros(self.resolution, dtype=bool)
        if self.env.nb_obstacles == 0:
            return binary_map
        indices = self.convert_obstacles_to_indices()
        binary_map[indices[:, 0], indices[:, 1]] = 1
        return binary_map

    def convert_obstacles_to_indices(self) -> np.ndarray:
        indices = []
        for obstacle in self.env.obstacles:
            pixel_min = np.floor(obstacle.min / (self.steps * self.env.pixel)).astype(
                int
            )
            pixel_max = np.ceil(obstacle.max / (self.steps * self.env.pixel)).astype(
                int
            )
            indices.extend(
                [
                    (i, j)
                    for i in range(pixel_min[0], pixel_max[0])
                    for j in range(pixel_min[1], pixel_max[1])
                ]
            )
        return np.array(indices)

    def convert_goal_to_binary_map(self, shape: str) -> np.ndarray:
        assert (
            self.env.goal is not None
        ), "Environment does not contain obstacles. Reset environment first."
        assert shape.lower() in [
            "box",
            "point",
        ], "Shape must be either 'box' or 'point'"

        binary_map = np.zeros(self.resolution, dtype=bool)
        indices = self.convert_goal_to_indices(shape)
        binary_map[indices[:, 0], indices[:, 1]] = 1
        return binary_map

    def convert_goal_to_indices(self, shape: str) -> np.ndarray:
        assert (
            self.env.goal is not None
        ), "Environment does not contain obstacles. Reset environment first."
        assert shape.lower() in [
            "box",
            "point",
        ], "Shape must be either 'box' or 'point'"

        if shape.lower() == "box":
            pixel_min = np.floor(
                self.env.goal.min / (self.steps * self.env.pixel)
            ).astype(int)
            pixel_max = np.ceil(
                self.env.goal.max / (self.steps * self.env.pixel)
            ).astype(int)
            return np.array(
                [
                    (i, j)
                    for i in range(pixel_min[0], pixel_max[0])
                    for j in range(pixel_min[1], pixel_max[1])
                ]
            )
        elif shape.lower() == "point":
            pixel = np.floor(
                self.env.goal.center / (self.steps * self.env.pixel)
            ).astype(int)
            return np.array([pixel])
        else:
            raise ValueError("Shape must be either 'box' or 'point'")

    def convert_free_positions_to_point_cloud(self, step: int = 1) -> np.ndarray:
        """Transform free space of the environment into a point cloud.

        Args:
            step (int): Step between 2 positions.
        Returns:
            np.ndarray: free positions as a point cloud [N, 2]
        """
        assert isinstance(step, int) and step > 0, "Step must be an integer > 0"
        positions = []
        for x, y in self.env.free_positions:
            if x % step == 0 and y % step == 0:
                positions.append((x + 0.5, y + 0.5))
        return np.array(positions)

    def get_agent_position(self, repeats: int = 1) -> np.ndarray:
        pixel = self.env.pixel if self.dim == 2 else 1
        center = np.array(self.env.agent.position.center) / pixel
        return np.repeat(center[None], axis=0, repeats=repeats)

    def get_goal_position(self) -> np.ndarray:
        pixel = self.env.pixel if self.dim == 2 else 1
        return np.array(self.env.goal.center) / pixel


class Env3DConverter(EnvConverter):
    # def __init__(
    #     self, env: GridWorld3D, resolution: Optional[Union[int, Tuple[int]]] = None
    # ):
    #     assert isinstance(
    #         env, GridWorld3D
    #     ), f"Environmnent must be of type GridWorld3D, not {type(env)}"
    #     self.env = env
    #     self.dim = 3
    #     # Set the resolution of the converter
    #     if resolution is None:
    #         resolution = (env.height, env.width, env.depth)
    #     self.resolution = resolution

    ## Public methods
    def convert_obstacles_to_binary_map(self) -> np.ndarray:
        assert (
            self.env.obstacles is not None
        ), "Environment does not contain obstacles. Reset environment first."

        binary_map = np.zeros(self.resolution, dtype=bool)
        if self.env.nb_obstacles == 0:
            return binary_map
        indices = self.convert_obstacles_to_indices()
        binary_map[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        return binary_map
        # X, Y, Z = self.meshgrid

        # for obstacle in self.env.obstacles:
        #     binary_map[
        #         (X >= obstacle.x)
        #         & (X < obstacle.max[0])
        #         & (Y >= obstacle.y)
        #         & (Y < obstacle.max[1])
        #         & (Z >= obstacle.z)
        #         & (Z < obstacle.max[2])
        #     ] = 1
        # return binary_map

    def convert_obstacles_to_indices(self) -> np.ndarray:
        indices = []
        for obstacle in self.env.obstacles:
            pixel_min = np.floor(obstacle.min / self.steps).astype(int)
            pixel_max = np.ceil(obstacle.max / self.steps).astype(int)
            indices.extend(
                [
                    (i, j, k)
                    for i in range(pixel_min[0], pixel_max[0])
                    for j in range(pixel_min[1], pixel_max[1])
                    for k in range(pixel_min[2], pixel_max[2])
                ]
            )
        return np.array(indices)

    def convert_goal_to_binary_map(self, shape: str = "box") -> np.ndarray:
        assert (
            self.env.goal is not None
        ), "Environment does not contain obstacles. Reset environment first."
        assert shape.lower() in [
            "box",
            "point",
        ], "Shape must be either 'box' or 'point'"

        binary_map = np.zeros(self.resolution, dtype=bool)
        indices = self.convert_goal_to_indices(shape)
        binary_map[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        return binary_map
        # if shape == "box":
        #     X, Y, Z = self.meshgrid
        #     binary_map[
        #         (X >= self.env.goal.x)
        #         & (X < self.env.goal.max[0])
        #         & (Y >= self.env.goal.y)
        #         & (Y < self.env.goal.max[1])
        #         & (Z >= self.env.goal.z)
        #         & (Z < self.env.goal.max[2])
        #     ] = 1
        # elif shape == "point":
        #     x, y, z = self.linspace
        #     ind_x, ind_y, ind_z = (
        #         np.argmin(np.abs(x - self.env.goal.x)),
        #         np.argmin(np.abs(y - self.env.goal.y)),
        #         np.argmin(np.abs(z - self.env.goal.z)),
        #     )
        #     binary_map[ind_x, ind_y, ind_z] = 1
        # else:
        #     raise ValueError("Shape must be either 'box' or 'point'")

        # return binary_map

    def convert_goal_to_indices(self, shape: str = "box") -> np.ndarray:
        assert (
            self.env.goal is not None
        ), "Environment does not contain obstacles. Reset environment first."
        assert shape.lower() in [
            "box",
            "point",
        ], "Shape must be either 'box' or 'point'"

        if shape.lower() == "box":
            pixel_min = np.floor(self.env.goal.min / self.steps).astype(int)
            pixel_max = np.ceil(self.env.goal.max / self.steps).astype(int)
            return np.array(
                [
                    (i, j, k)
                    for i in range(pixel_min[0], pixel_max[0])
                    for j in range(pixel_min[1], pixel_max[1])
                    for k in range(pixel_min[2], pixel_max[2])
                ]
            )
        elif shape.lower() == "point":
            pixel = np.floor(self.env.goal.center / self.steps).astype(int)
            return np.array([pixel])
        else:
            raise ValueError("Shape must be either 'box' or 'point'")

    def convert_free_positions_to_point_cloud(self, step: int = 1) -> np.ndarray:
        assert isinstance(step, int) and step > 0, "Step must be an integer > 0"
        positions = []
        for x, y, z in self.env.free_positions:
            if x % step == 0 and y % step == 0 and z % step == 0:
                positions.append((x + 0.5, y + 0.5, z + 0.5))
        return np.array(positions)

    def get_agent_position(self, repeats: int = 1) -> np.ndarray:
        return np.repeat(self.env.agent.center[None], axis=0, repeats=repeats)

    def get_goal_position(self) -> np.ndarray:
        return self.env.goal.center

    ## Properties


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from snake_ai.envs.random_obstacles_3d import RandomObstacles3D
    from snake_ai.envs.random_obstacles_env import RandomObstaclesEnv

    # bounds = (10, 10)
    # obstacles = [Rectangle(0, 0, 1, 1), Rectangle(9, 0, 1, 1), Rectangle(0, 9, 1, 1)]
    # b_map = test_2d(obstacles, 40, bounds)
    # print(np.argwhere(b_map))
    # plt.imshow(b_map, extent=[0, 10, 10, 0])
    # plt.show()

    N = 2

    env = RandomObstacles3D(10, 10, 10, nb_obs=10, max_size=1)
    env_2d = RandomObstaclesEnv(10, 10, nb_obs=10, max_size=1)
    env.reset()
    env_2d.reset()
    # env.obstacles = [
    #     Cube(0, 0, 0, 1, 1, 1),
    #     Cube(9, 0, 0, 1, 1, 1),
    #     Cube(0, 9, 0, 1, 1, 1),
    #     Cube(0, 0, 9, 1, 1, 1),
    # ]
    converter = Env3DConverter(env, 1)
    converter_2d = Env2DConverter(env_2d, 10)

    bmap = converter_2d.convert_obstacles_to_binary_map()
    print(bmap.shape)
    plt.imshow(bmap, extent=[0, 10, 10, 0])
    point_cloud = converter_2d.convert_free_positions_to_point_cloud()
    plt.scatter(point_cloud[:, 1], point_cloud[:, 0], c="r")
    plt.show()

    # binary_map = converter.convert_obstacles_to_binary_map()
    binary_map = converter.convert_goal_to_binary_map("point")
    # ind = converter.convert_obstacles_to_indices()
    # binary_map[ind[:, 0], ind[:, 1], ind[:, 2]] = 1
    binary_map = converter.convert_obstacles_to_binary_map()
    print(binary_map.shape)

    ax = plt.figure().add_subplot(projection="3d")
    ax.voxels(binary_map)
    point_cloud = converter.convert_free_positions_to_point_cloud()
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c="r")
    print(env.obstacles)
    print(np.argwhere(binary_map))

    # fig, ax = plt.subplots(N, N)
    # for i in range(N):
    #     for j in range(N):
    #         z = i + N * j
    #         ax[i, j].imshow(binary_map[:, :, z])
    #         ax[i, j].set(title=f"z= {z}", xlabel="y", ylabel="x")
    plt.show()
