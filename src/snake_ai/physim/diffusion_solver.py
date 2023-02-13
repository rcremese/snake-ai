##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Solver 2D of the diffusion equation using Finite Element Methods
 # @desc Created on 2023-02-04 7:48:44 am
 # @copyright https://mit-license.org/
 #
import matplotlib.pyplot as plt
from snake_ai.envs.geometry import Geometry, Geometry
from phi.jax import flow
import numpy as np
from typing import List, Tuple, Optional
from snake_ai.utils.types import Numerical
from pathlib import Path

@flow.math.jit_compile
def explicit_diffusion(concentration : flow.field.Field, diffusion : Numerical, dt : Numerical) -> flow.field.Field:
    return flow.diffuse.explicit(concentration, diffusivity=diffusion, dt=dt)

@flow.math.jit_compile
def explicit_diffusion_with_obstacles(concentration : flow.field.Field, obstacle_mask : flow.field.Field, diffusion : Numerical, dt : Numerical) -> flow.field.Field:
    return (1-obstacle_mask) * flow.diffuse.explicit(concentration, diffusivity=diffusion, dt=dt)

@flow.math.jit_compile
def crank_nicolson_diffusion(concentration : flow.field.Field, diffusion : Numerical, dt : Numerical) -> flow.field.Field:
    return flow.diffuse.implicit(concentration, diffusivity=diffusion, dt=dt, order=2)

@flow.math.jit_compile
def crank_nicolson_diffusion_with_obstacles(concentration : flow.field.Field, obstacle_mask : flow.field.Field, diffusion : Numerical, dt : Numerical) -> flow.field.Field:
    return (1-obstacle_mask) * flow.diffuse.implicit(concentration, diffusivity=diffusion, dt=dt, order=2)

class DiffusionSolver2D:
    def __init__(self, x_max : Numerical, y_max : Numerical, t_max : Numerical, source: Geometry, init_value : float = 1, diff_coef : float = 1,
                 obstacles: List[Geometry] = None, grid_res : int or Tuple[int] = None) -> None:
        # Bounds
        if not (isinstance(x_max, (float, int)) or isinstance(y_max, (float, int))):
            raise ValueError(f"Expected values for simulation bounds must be float or int, not {type(x_max), type(y_max)}")
        self._x_max = abs(x_max)
        self._y_max = abs(y_max)
        # Tmax
        if not isinstance(t_max, (float, int)) or t_max <= 0:
            raise ValueError(f"Simulation ending time need to be a positive int or float, not {t_max}")
        self.t_max = t_max
        # Obstacles
        if obstacles is None:
            obstacles = []
        assert all([isinstance(obs, Geometry) for obs in obstacles]), "Only instances of Obstacle are allowed."
        self.obstacles = [obstacle.to_phiflow() for obstacle in obstacles] # Creates a phiflow mask array from a list of obstacles
        # Initial distrib
        if not isinstance(source, Geometry):
            raise TypeError(f"Initial distribution need be an instance of Rectangle, not {type(source)}")
        if source.center[0] > self._x_max or source.center[1] > self._y_max:
            raise ValueError(f"Source center can not be out of bounds ([0,{self._x_max}], [0,{self._y_max}]). Get {source.center}")
        self._source = source.to_phiflow()
        # Initial value & diffusion coefficient
        if init_value <= 0:
            raise ValueError(f"The initial value can not be negative. Get {init_value}")
        self._init_value = init_value
        if diff_coef <= 0:
            raise ValueError(f"The diffusion coefficient can not be negative. Get {init_value}")
        self._diff_coef = diff_coef
        # Grid resolution
        if grid_res is None:
            grid_res = (x_max, y_max)
        if isinstance(grid_res, int):
            grid_res = (grid_res, grid_res)
        if len(grid_res) != 2:
            raise ValueError(f"Too many values to unpack. Grid resolution expect at most 2 values (x_res, y_res). Get {len(grid_res)}")
        if any(res <= 0 for res in grid_res):
            raise ValueError(f"Grid resolution values can not be <= 0. Get {grid_res}")
        self._grid_res = {'x' : grid_res[0], 'y' : grid_res[1]}
        # Defined properties
        self.dt = None
        self.concentration = None
        self.obstacle_mask = None
        # Initialise concentration and obstacle mask
        self.reset()

    def reset(self):
        bounds = flow.Box(x=self._x_max, y=self._y_max)
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

    def step(self, concentration) -> flow.CenteredGrid:
        if self.obstacle_mask is None:
            return crank_nicolson_diffusion(concentration, self._diff_coef, self.dt)
        else:
            return crank_nicolson_diffusion_with_obstacles(concentration, self.obstacle_mask, self._diff_coef, self.dt)

    def start(self, nb_samples : int = 0) -> flow.CenteredGrid:
        assert (nb_samples >= 0), f"The value for which we want to save the solution must be positive. Get {nb_samples}"
        return_history = nb_samples > 1
        t = 0
        if return_history:
            time_samples_iter = iter(np.linspace(0, self.t_max, nb_samples))
            time_sample = next(time_samples_iter)
            # As we save the initial concentration at the beginning
            history = [self.concentration]
            nb_samples -= 1
            time_sample = next(time_samples_iter)

        while t < self.t_max:
            self.concentration = self.step(self.concentration)
            t += self.dt
            print(t)
            # Save the history of the diffusion equation
            if return_history and (t >= time_sample):
                history.append(self.concentration)
                nb_samples -= 1
                try:
                    time_sample = next(time_samples_iter)
                except StopIteration:
                    break

        # In case we store an history, that's what is returned
        if return_history:
            return flow.field.stack(history, flow.batch('time'))
        return self.concentration

    def write(self, output_path : Path or str):
        dirpath = Path(output_path).resolve(strict=True)
        filename = f"diffusion_Tmax={self.t_max}_D={self._diff_coef}"

    def __repr__(self) -> str:
        return f"{__class__.__name__}(x_max={self._x_max}, y_max={self._y_max}, t_max={self.t_max}, diff_coef={self._diff_coef},\
            source={self._source!r}, init_value={self._init_value}, obstacles={self.obstacles!r}, grid_res={self._grid_res!r})"

def main():
    from snake_ai.envs import SnakeClassicEnv
    env = SnakeClassicEnv(width=10, height=10,nb_obstacles=5, pixel=10)
    env.reset()
    x_max, y_max = env.window_size
    t_max = 100
    diff = 1
    init = 1_000
    diff_solver = DiffusionSolver2D(x_max, y_max, t_max, source=env.food, init_value=init,diff_coef=diff, obstacles=env.obstacles)
    solutions = diff_solver.start(nb_samples=1)

    logs = flow.math.log(flow.math.where(solutions.values < 1e-4, 1e-4, solutions.values))
    grads = flow.math.spatial_gradient(logs, padding=flow.extrapolation.ONE * np.log(1e-4))
    flow.vis.plot([solutions, logs, grads], show_color_bar=False, cmap='inferno')
    plt.show()

if __name__ == '__main__':
    main()