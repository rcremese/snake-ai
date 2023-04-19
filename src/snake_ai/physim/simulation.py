##
# @author Robin CREMESE <robin.cremese@gmail.com>
# @file Meta class used by the user to create a simulation of physical process
# @desc Created on 2022-12-13 4:35:40 pm
# @copyright https://mit-license.org/
#
from abc import ABC, abstractmethod
# from snake_ai.physim import DiffusionSolver2D
from snake_ai.physim.solver import DiffusionSolver
from snake_ai.physim.converter import DiffusionConverter, PointCloudConverter, ObstacleConverter
from snake_ai.envs import GridWorld, SnakeClassicEnv
from snake_ai.utils import errors
from snake_ai.utils.types import Numerical
from typing import Union, Optional, List, Any, Dict
from phi.jax import flow
from pathlib import Path
import numpy as np
import json

MAX_TIME = 1
HISTORY_LENGTH = 100
class Simulation(ABC):
    resolutions = ["pixel", "meta"]
    def __init__(self, env : GridWorld, res : str = "pixel", init_value : float = 1, t_max : Optional[float] = None, dt : Optional[float] = None, history : bool = False, **kwargs):
        # Set the environment
        if not isinstance(env, GridWorld):
            raise TypeError(f"Environment need to be a GridWorld. Get {type(env)} instead")
        self.env = env
        # Set the resolution of the simulation ('pixel' or 'meta-pixel')
        if res.lower() not in self.resolutions:
            raise ValueError(f"Resolution need to be in {self.resolutions}. Get {res} instead")
        self.res = res.lower()
        # Set the initial value of the field
        if init_value <= 0:
            raise ValueError(f"The initial value need to be > 0. Get {init_value}")
        self._init_value = init_value
        # Set the maximum simulation time
        if (t_max is not None) and (t_max <= 0):
            raise ValueError(f"The maximum simulation time need to be > 0. Get {t_max} instead")
        self.t_max = t_max
        # Set the time step
        if (dt is not None) and (dt <= 0):
            raise ValueError(f"The simulation time step need to be > 0. Get {dt} instead")
        self.dt = dt
        # Set the history step, in order to have a history of HISTORY_LENGTH steps
        self._hstep = self.t_max / HISTORY_LENGTH if history else 0

        # Converters
        self._obstacles_converter = ObstacleConverter(self.res)
        self._point_cloud_converter = PointCloudConverter(self.res)
        # Attributes to be initialised
        self._field_converter = None
        self._solver = None
        self._field = None

    ##Public methods
    @abstractmethod
    def start(self):
        raise NotImplementedError()

    def reset(self, seed : Optional[int] = None):
        self.env.reset(seed)
        # Mask the field with the obstacles and set the initial value
        self._field = self._init_value * (1 - self.obstacles) * self._field_converter(self.env)

    ## Properties
    @property
    def field(self):
        if self._field is None:
            raise errors.InitialisationError("The field is not initialised. Use the reset method of the simulation first")
        return self._field

    @field.setter
    def field(self, new_field : flow.CenteredGrid):
        if not isinstance(new_field, flow.CenteredGrid):
            raise TypeError(f"The field need to be a CenteredGrid. Get {type(new_field)} instead")
        self._field = new_field

    @property
    def obstacles(self):
        if self.env._obstacles is None:
            raise errors.InitialisationError("The environment is not initialised. Use the reset method of the simulation first")
        return self._obstacles_converter(self.env)

    @property
    def point_cloud(self):
        if self.env._obstacles is None:
            raise errors.InitialisationError("The environment not initialised. Use the reset method of the simulation first")
        return self._point_cloud_converter(self.env)

    @property
    def history(self):
        if self._hstep == 0:
            raise ValueError("The history is not recorded. Set history_step to a positive value to record the history.")
        return self._solver.history

    @property
    def hparams(self) -> Dict[str, Any]:
        return {
            "res" : self.res,
            "t_max" : self.t_max,
            "dt" : self.dt,
        }

class DiffusionSimulation(Simulation):
    solvers = ["explicit", "implicit", "crank_nicolson"]
    def __init__(self, env : GridWorld, res : str = "pixel", init_value : float = 1, t_max : Optional[float] = None,  dt : Optional[float] = None,
                 history : bool = False, diffusivity : float = 1,  solver = "crank_nicolson", stationary : bool = False, **kwargs):
        super().__init__(env, res, init_value, t_max, dt, history)
        # Set the field diffusive coefficient
        if diffusivity <= 0 :
            raise ValueError(f"The diffusion coefficient needs to be > 0. Get {diffusivity}")
        # Set the stopping criteria
        area = self.env.width * self.env.height if self.res == "meta" else self.env.width * self.env.height * self.env.pixel ** 2
        if self.t_max is None:
            self.t_max = MAX_TIME * area / diffusivity # Stop condition based on the diffusion time needed to diffuse in a free environment
        # Set the time step of the scheme, considering the resolution to be 1 in each environment
        if self.dt is not None:
            if res == "meta":
                dt = 1
            else:
                dt = self.env.pixel
            if solver == "explicit":
                dt /= (2 * diffusivity) # Stability condition for the explicit scheme

        self._solver = DiffusionSolver(diffusivity, self.t_max, dt=self.dt, history_step=self._hstep, name=solver, stationary=stationary)
        self._field_converter = DiffusionConverter(self.res)

    # def reset(self, seed : Optional[int] = None):
    #     self.env.reset(seed)
    #     # Initialise the field, the obstacles and the point cloud
    #     obs_converter = ObstacleConverter(self.res)
    #     self._obstacles = obs_converter(self.env)
    #     field_converter = DiffusionConverter(self.res)
    #     # Mask the field with the obstacles and set the initial value
    #     self._field = self._init_value * (1 - self._obstacles) * field_converter(self.env)
    #     point_cloud_converter = PointCloudConverter(self.res)
    #     self._point_cloud = point_cloud_converter(self.env)

    def start(self):
        if self._field is None:
            raise errors.InitialisationError("The field is not initialised. Use the reset method before")
        self._field = self._solver.solve(self._field, self.obstacles)

    # @property
    # def field(self):
    #     if self._field is None:
    #         raise errors.InitialisationError("The field is not initialised. Use the reset method before")
    #     return self._field

    # @field.setter
    # def field(self, new_field : flow.CenteredGrid):
    #     if not isinstance(new_field, flow.CenteredGrid):
    #         raise TypeError(f"The field need to be a CenteredGrid. Get {type(new_field)} instead")
    #     self._field = new_field

    # @property
    # def obstacles(self):
    #     if self._obstacles is None:
    #         raise errors.InitialisationError("The obstacles are not initialised. Use the reset method before")
    #     return self._obstacles

    # @property
    # def point_cloud(self):
    #     if self._point_cloud is None:
    #         raise errors.InitialisationError("The point cloud is not initialised. Use the reset method before")
    #     return self._point_cloud

    # @property
    # def history(self):
    #     if self._hstep == 0:
    #         raise ValueError("The history is not recorded. Set history_step to a positive value to record the history.")
    #     return self._solver.history

    @Simulation.hparams.getter
    def hparams(self) -> Dict[str, Any]:
        hparams = super().hparams
        hparams["diffusivity"] = self._solver.diffusivity
        hparams["init_value"] = self._init_value
        hparams["solver"] = self._solver.name
        hparams["stationary"] = self._solver.is_stationary
        return hparams

if __name__ == "__main__":
    from snake_ai.envs import MazeGrid, RandomObstaclesEnv
    import matplotlib.pyplot as plt

    env = RandomObstaclesEnv(nb_obs=10, max_obs_size=3)
    simulation = DiffusionSimulation(env, res="pixel", diffusivity=1, init_value=1e6, t_max=1000, dt=1, stationary=True)
    simulation.reset()
    simulation.start()
    flow.vis.plot(flow.math.log(simulation.field))
    plt.show()
## TODO : Nettoyer le code
# INIT_VALUE = 1e6
# class Simulation2:
#     def __init__(self, width : int, height : int, nb_obs : int, max_size : int, pixel : int, seed : int, diff: Numerical, t_max : Numerical) -> None:
#         # Initialise the environment and seed it
#         self.env = SnakeClassicEnv(render_mode=None, width=width, height=height, nb_obstacles=nb_obs, pixel=pixel, max_obs_size=max_size, seed=seed)
#         if not isinstance(seed, int):
#             raise TypeError(f"Seed need to be an int. Get {type(seed)}")
#         self._seed = seed

#         if t_max <= 0 or diff <= 0:
#             raise ValueError(f"The diffusion coefficient and the maximum simulation time need to be > 0. Get {diff} and {t_max} instead")
#         self.t_max = t_max
#         self.diff = diff
#         # Arguments to be instanciated
#         self.diffusion_solver = None

#     def reset(self):
#         self.env.reset()
#         x_max, y_max = self.env.window_size
#         self.diffusion_solver = DiffusionSolver2D(x_max, y_max, self.t_max, source=self.env.food, init_value=INIT_VALUE, obstacles=self.env.obstacles, diff_coef=self.diff)

#     def write(self, dirpath : Union[str, Path]):
#         dirpath = Path(dirpath).resolve(strict=True)
#         dirpath = dirpath.joinpath(self.name)
#         dirpath.mkdir()

#         self.env.write(dirpath.joinpath("environment.json"), detailed=True)
#         flow.field.write(self.concentration, str(dirpath.joinpath("concentration_field")))
#         # Write the physical param in a specific file
#         phi_param = {"tmax" : self.t_max, "diffusion" : self.diff, "time" : self.diffusion_solver.time, "dt" : float(self.diffusion_solver.dt)}
#         with open(dirpath.joinpath("physical_params.json"), "w") as file:
#             json.dump(phi_param, file)

#     @classmethod
#     def load(cls, dirpath : Union[str, Path]):
#         dirpath = Path(dirpath).resolve(strict=True)
#         for filename in ["environment.json", "physical_params.json", "concentration_field.npz"]:
#             assert dirpath.joinpath(filename).exists(), f"Directory {dirpath.name} does not contain the file {filename} needed to load the simulation"
#         # Load the environment and physical param
#         env = SnakeClassicEnv.load(dirpath.joinpath("environment.json"))
#         with open(dirpath.joinpath("physical_params.json"), 'r') as file:
#             phy_param = json.load(file)
#         # Instanciate the simulation
#         simulation = cls(env.width, env.height, env.nb_obstacles, env._max_obs_size, env.pixel_size, env._seed, phy_param['diffusion'], phy_param['tmax'])
#         simulation.reset()
#         simulation.env = env
#         # Set the parameter of the diffusion solver
#         simulation.diffusion_solver.time = phy_param["time"]
#         simulation.diffusion_solver.concentration = flow.field.read(str(dirpath.joinpath("concentration_field.npz")))
#         return simulation

#     def get_valid_point_cloud(self) -> flow.PointCloud:
#         free_pos = np.array(self.env.free_positions, dtype=float) # array of free position in term of pixel size
#         centers = self.env.pixel_size * (free_pos + 0.5)
#         points = []
#         for center in centers:
#             points.append(flow.vec(x=center[0], y=center[1]))
#         points = flow.tensor(points, flow.instance('point'))
#         return flow.PointCloud(points, points, bounds=self.concentration.bounds)

#     @staticmethod
#     def get_gradient(field : flow.Grid) -> flow.CenteredGrid:
#         flow.field.spatial_gradient(field)

#     def start(self):
#         if self.diffusion_solver is None:
#             self.reset()
#         self.diffusion_solver.start()

#     @property
#     def concentration(self) -> flow.CenteredGrid:
#         "Concentration field at time Tmax in the given environment"
#         if self.diffusion_solver is None:
#             self.reset()
#         return self.diffusion_solver.concentration

#     @property
#     def name(self) -> str:
#         "Name used to identify the simulation"
#         x_max, y_max = self.env.window_size
#         return f"diffusion_Tmax={self.t_max}_D={self.diff}_Nobs={len(self.env.obstacles)}_size={self.env._max_obs_size}_Box({x_max},{y_max})_seed={self._seed}"

#     def __repr__(self) -> str:
#         return f"{__class__.__name__}(env={self.env!r}, solver={self.diffusion_solver!r})"
