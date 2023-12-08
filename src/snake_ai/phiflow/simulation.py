##
# @author Robin CREMESE <robin.cremese@gmail.com>
# @file Meta class used by the user to create a simulation of physical process
# @desc Created on 2022-12-13 4:35:40 pm
# @copyright https://mit-license.org/
#
from abc import ABC, abstractmethod

# from snake_ai.physim import DiffusionSolver2D
from snake_ai.phiflow.solver import DiffusionSolver
from snake_ai.phiflow.converter import (
    DiffusionConverter,
    PointCloudConverter,
    ObstacleConverter,
)
from snake_ai.envs import GridWorld
from snake_ai.utils import errors
from typing import Union, Optional, List, Any, Dict
from phi.jax import flow
from enum import Enum


class TimeFactor(Enum):
    """
    Time factor used to scale the simulation time
    """

    GridWorld = 1
    MazeGrid = 10
    RandomObstaclesEnv = 2
    RoomEscape = 4
    SlotEnv = 2


HISTORY_LENGTH = 100


class Simulation(ABC):
    resolutions = ["pixel", "meta"]

    def __init__(
        self,
        env: GridWorld,
        res: str = "pixel",
        init_value: float = 1,
        t_max: Optional[float] = None,
        dt: Optional[float] = None,
        history: bool = False,
        **kwargs,
    ):
        # Set the environment
        if not isinstance(env, GridWorld):
            raise TypeError(
                f"Environment need to be a GridWorld. Get {type(env)} instead"
            )
        self.env = env
        # Set the resolution of the simulation ('pixel' or 'meta-pixel')
        if res.lower() not in self.resolutions:
            raise ValueError(
                f"Resolution need to be in {self.resolutions}. Get {res} instead"
            )
        self.res = res.lower()
        # Set the initial value of the field
        if init_value <= 0:
            raise ValueError(f"The initial value need to be > 0. Get {init_value}")
        self._init_value = init_value
        # Set the maximum simulation time
        if (t_max is not None) and (t_max <= 0):
            raise ValueError(
                f"The maximum simulation time need to be > 0. Get {t_max} instead"
            )
        self.t_max = t_max
        # Set the time step
        if (dt is not None) and (dt <= 0):
            raise ValueError(
                f"The simulation time step need to be > 0. Get {dt} instead"
            )
        self.dt = dt
        # Set the history step, in order to have a history of HISTORY_LENGTH steps
        self._hstep = self.t_max / HISTORY_LENGTH if history else 0

        # Converters
        self._obstacles_converter = ObstacleConverter(self.res)
        self._point_cloud_converter = PointCloudConverter(self.res)
        # Attributes to be initialised
        # self._field_converter = None
        self._solver = None
        self._field = None

    ##Public methods
    @abstractmethod
    def start(self):
        raise NotImplementedError()

    def reset(self, seed: Optional[int] = None):
        self.env.reset(seed)
        # Mask the field with the obstacles and set the initial value
        # self._field = self._init_value * (1 - self.obstacles) * self._field_converter(self.env)

    ## Properties
    @property
    def field(self):
        if self._field is None:
            raise errors.InitialisationError(
                "The field is not initialised. Use the reset method of the simulation first"
            )
        return self._field

    @field.setter
    def field(self, new_field: flow.CenteredGrid):
        if not isinstance(new_field, flow.CenteredGrid):
            raise TypeError(
                f"The field need to be a CenteredGrid. Get {type(new_field)} instead"
            )
        self._field = new_field

    @property
    def obstacles(self):
        if self.env._obstacles is None:
            raise errors.InitialisationError(
                "The environment is not initialised. Use the reset method of the simulation first"
            )
        return self._obstacles_converter(self.env)

    @property
    def point_cloud(self):
        if self.env._obstacles is None:
            raise errors.InitialisationError(
                "The environment not initialised. Use the reset method of the simulation first"
            )
        return self._point_cloud_converter(self.env)

    @property
    def history(self):
        if self._hstep == 0:
            raise ValueError(
                "The history is not recorded. Set history_step to a positive value to record the history."
            )
        return self._solver.history

    @property
    def hparams(self) -> Dict[str, Any]:
        return {
            "res": self.res,
            "t_max": self.t_max,
            "dt": self.dt,
        }


class DiffusionSimulation(Simulation):
    solvers = ["explicit", "implicit", "crank_nicolson"]
    resolutions = ["pixel", "meta"]

    def __init__(
        self,
        env: GridWorld,
        res: str = "pixel",
        init_value: float = 1,
        t_max: Optional[float] = None,
        dt: Optional[float] = None,
        history: bool = False,
        diffusivity: float = 1,
        solver="crank_nicolson",
        stationary: bool = False,
        **kwargs,
    ):
        super().__init__(env, res, init_value, t_max, dt, history)
        # Set the spatial resolution of the simulation
        spatial_res = 1 / self.env.pixel if self.res == "pixel" else 1
        # Set the field diffusive coefficient
        if diffusivity <= 0:
            raise ValueError(
                f"The diffusion coefficient needs to be > 0. Get {diffusivity}"
            )
        self._diffusivity = diffusivity
        # Set the stopping criteria
        area = self.env.width * self.env.height
        if self.t_max is None:
            # Stop condition based on the diffusion time needed to diffuse in a free environment
            self.t_max = (
                TimeFactor[self.env.__class__.__name__].value * area / diffusivity
            )
        # Set the time step of the scheme, considering the resolution to be 1 in each environment
        if self.dt is None:
            if solver == "explicit":
                # Stability condition for the explicit scheme
                self.dt = 0.25 * spatial_res**2 / self._diffusivity
            else:
                self.dt = spatial_res
        self._solver = DiffusionSolver(
            self._diffusivity,
            t_max=self.t_max,
            dt=self.dt,
            history_step=self._hstep,
            name=solver,
            stationary=stationary,
        )
        self._field_converter = DiffusionConverter(self.res)

    def start(self):
        if self._field is None:
            raise errors.InitialisationError(
                "The field is not initialised. Use the reset method before"
            )
        self._field = self._solver.solve(self._field, self.obstacles)

    def reset(self, seed: Optional[int] = None):
        super().reset(seed)
        self._field = (
            self._init_value * (1 - self.obstacles) * self._field_converter(self.env)
        )

    @property
    def name(self) -> str:
        return f"{self.env.name}_{self.res}_Tmax={self.t_max}_D={self._diffusivity}"

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
    simulation = DiffusionSimulation(
        env,
        res="pixel",
        diffusivity=1,
        init_value=1e6,
        dt=1,
        stationary=True,
    )
    simulation.reset()
    simulation.start()
    flow.vis.plot(flow.math.log(simulation.field))
    plt.show()
