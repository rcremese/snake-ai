from snake_ai.envs import SnakeClassicEnv, Circle
from phi.jax import flow

from abc import ABCMeta, abstractmethod
from snake_ai.utils.errors import EmptyEnvironmentError
from typing import Optional, Tuple

class Converter(metaclass=ABCMeta):
    def __init__(self, size : str) -> None:
        if not size.lower() in ['pixel', 'meta']:
            raise ValueError(f"The size argument must be 'pixel' or 'meta', not {size}")
        self._size = size.lower()

    @abstractmethod
    def convert(self, env) -> Tuple[flow.CenteredGrid, flow.Geometry]:
        raise NotImplementedError()

    @abstractmethod
    def _convert_pixel(self, env):
        raise NotImplementedError()

    @abstractmethod
    def _convert_metapixel(self, env):
        raise NotImplementedError()

class DiffusionConverter(Converter):
    def __init__(self, size: str, init_value : float) -> None:
        super().__init__(size)
        if init_value <= 0:
            raise ValueError("Initial value for diffusion process must be > 0.")
        self._init_value = init_value

    def convert(self, env : SnakeClassicEnv) -> Tuple[flow.CenteredGrid, Optional[flow.Geometry]]:
        assert isinstance(env, SnakeClassicEnv), f"Expected instance of SnakeClassicEnv, get {type(env)}"
        if (env.food is None) or (env.obstacles is None):
            raise EmptyEnvironmentError(f"The environment {env} does not have food and obstacles initialized")

        if self._size == "pixel":
            return self._convert_pixel(env)
        else:
            return self._convert_metapixel(env)

    def _convert_metapixel(env : SnakeClassicEnv) -> Tuple[flow.CenteredGrid, Optional[flow.Geometry]]:
        bounds = flow.Box(x=env.width, y=env.height)
        source =  Circle(*env.food.center // env.pixel_size)

        # Define the initial concentration as a square with init_value in a grid which bounds are x_max and y_max.
        # Absorbing boundary conditions are set on the frontier and the obstacles
        self.concentration = self._init_value * flow.CenteredGrid(self._source, bounds=bounds, **self._grid_res)
        # As second order scheme in time is used, dt is set to be of order dx
        self.dt = min(self.concentration.dx)
        # Define a mask containing all the obstacles and of the same size as concentration.
        # Set to None if there is no obstacles in the list
        self.obstacle_mask = flow.HardGeometryMask(flow.union(self.obstacles)) @ self.concentration if self.obstacles else None
        if self.obstacle_mask is not None:
            self.concentration = (1 - self.obstacle_mask) * self.concentration

class PointCloudConverter(Converter):
    pass