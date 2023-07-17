##
# @author  <robin.cremese@gmail.com>
# @file Class to convert a GridWorld environment into a phiflow environment
# @desc Created on 2023-04-04 11:03:07 am
# @copyright MIT License
#
from snake_ai.envs import GridWorld, Rectangle
from phi.jax import flow

from abc import ABCMeta, abstractmethod
from snake_ai.utils import errors
from typing import Optional, Tuple, List, Union


def convert_obstacles(
    obstacles: List[Rectangle], div_factor: int = 1
) -> List[flow.Box]:
    # Handle the case where there is only one obstacle
    if isinstance(obstacles, Rectangle):
        obstacles = [obstacles]
    assert isinstance(obstacles, list) and all(
        isinstance(obs, Rectangle) for obs in obstacles
    ), f"Expected a list of Rectangle as input. Get {obstacles}"
    assert (
        isinstance(div_factor, int) and div_factor >= 1
    ), f"Expected div_factor > 0, get {div_factor}"

    return [
        flow.Box(
            x=(obs.left // div_factor, obs.right // div_factor),
            y=(obs.top // div_factor, obs.bottom // div_factor),
        )
        for obs in obstacles
    ]


class Converter(metaclass=ABCMeta):
    def __init__(self, type: str) -> None:
        if not type.lower() in ["pixel", "meta"]:
            raise ValueError(f"The type argument must be 'pixel' or 'meta', not {type}")
        self.type = type.lower()

    ## Private methods
    def _check_environment(self, env: GridWorld) -> None:
        if not isinstance(env, GridWorld):
            raise TypeError(f"Expected instance of GridWorld, get {type(env)}")
        if (env.goal is None) or (env.obstacles is None):
            raise errors.InitialisationError(
                f"The environment {env} does not have goal and obstacles initialized. Rest the environment !"
            )

    ## Dunder methods
    @abstractmethod
    def __call__(self, env: GridWorld) -> flow.field.SampledField:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{__class__.__name__}(size={self.type})"


class DiffusionConverter(Converter):
    ## Dunder methods
    def __call__(self, env: GridWorld) -> flow.CenteredGrid:
        self._check_environment(env)
        # Select the scaling factor from the type of conversion
        if self.type == "pixel":
            source = flow.Sphere(
                x=env.goal.centerx / env.pixel,
                y=env.goal.centery / env.pixel,
                radius=env.pixel // 2,
            )
        else:
            source = flow.Box(
                x=(env.goal.left // env.pixel, env.goal.right // env.pixel),
                y=(env.goal.top // env.pixel, env.goal.bottom // env.pixel),
            )

        bounds = flow.Box(x=env.width, y=env.height)
        return flow.CenteredGrid(source, bounds=bounds, x=env.width, y=env.height)

    def __repr__(self) -> str:
        return f"{__class__.__name__}(size={self.type})"


class ObstacleConverter(Converter):
    def __call__(self, env: GridWorld) -> flow.CenteredGrid:
        if self.type == "pixel":
            tensor = flow.spatial(x=env.window_size[0], y=env.window_size[1])
        else:
            tensor = flow.spatial(x=env.width, y=env.height)
        grid = flow.CenteredGrid(
            flow.math.zeros(tensor), bounds=flow.Box(x=env.width, y=env.height)
        )
        if env.obstacles == []:
            return grid
        obstacles = convert_obstacles(env.obstacles, env.pixel)
        return flow.field.resample(flow.union(obstacles), grid)


class PointCloudConverter(Converter):
    def __call__(self, env: GridWorld) -> flow.PointCloud:
        points = []
        for x, y in env.free_positions:
            points.append(flow.vec(x=(x + 0.5), y=(y + 0.5)))
        position = flow.tensor(points, flow.instance("walker"))
        velocity = flow.math.zeros_like(position)
        return flow.PointCloud(
            position, values=velocity, bounds=flow.Box(x=env.width, y=env.height)
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from snake_ai.envs import MazeGrid, RandomObstaclesEnv

    env = RandomObstaclesEnv(nb_obs=10)
    env.reset()
    converter = DiffusionConverter("meta")
    obstacles = ObstacleConverter("meta")
    pt_converter = PointCloudConverter("meta")

    init_distrib = converter(env)
    obs_mask = obstacles(env)
    pts = pt_converter(env)
    flow.vis.plot([init_distrib, obs_mask, pts])
    plt.show()
