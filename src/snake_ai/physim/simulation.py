##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-12-13 4:35:40 pm
 # @copyright https://mit-license.org/
 #
from snake_ai.physim import DiffusionSolver2D
from snake_ai.envs import SnakeClassicEnv
from snake_ai.utils.types import Numerical
from typing import Union, List
from phi.jax import flow
from pathlib import Path
import numpy as np
import json

INIT_VALUE = 1e6
class Simulation:
    def __init__(self, width : int, height : int, nb_obs : int, max_size : int, pixel : int, seed : int, diff: Numerical, t_max : Numerical) -> None:
        # Initialise the environment and seed it
        self.env = SnakeClassicEnv(render_mode=None, width=width, height=height, nb_obstacles=nb_obs, pixel=pixel, max_obs_size=max_size, seed=seed)
        if not isinstance(seed, int):
            raise TypeError(f"Seed need to be an int. Get {type(seed)}")
        self._seed = seed

        if t_max <= 0 or diff <= 0:
            raise ValueError(f"The diffusion coefficient and the maximum simulation time need to be > 0. Get {diff} and {t_max} instead")
        self.t_max = t_max
        self.diff = diff
        # Arguments to be instanciated
        self.diffusion_solver = None

    def reset(self):
        self.env.reset()
        x_max, y_max = self.env.window_size
        self.diffusion_solver = DiffusionSolver2D(x_max, y_max, self.t_max, source=self.env.food, init_value=INIT_VALUE, obstacles=self.env.obstacles, diff_coef=self.diff)

    def write(self, dirpath : Union[str, Path]):
        dirpath = Path(dirpath).resolve(strict=True)
        dirpath = dirpath.joinpath(self.name)
        dirpath.mkdir()

        self.env.write(dirpath.joinpath("environment.json"), detailed=True)
        flow.field.write(self.concentration, str(dirpath.joinpath("concentration_field")))
        # Write the physical param in a specific file
        phi_param = {"tmax" : self.t_max, "diffusion" : self.diff, "time" : self.diffusion_solver.time, "dt" : float(self.diffusion_solver.dt)}
        with open(dirpath.joinpath("physical_params.json"), "w") as file:
            json.dump(phi_param, file)

    @classmethod
    def load(cls, dirpath : Union[str, Path]):
        dirpath = Path(dirpath).resolve(strict=True)
        for filename in ["environment.json", "physical_params.json", "concentration_field.npz"]:
            assert dirpath.joinpath(filename).exists(), f"Directory {dirpath.name} does not contain the file {filename} needed to load the simulation"
        # Load the environment and physical param
        env = SnakeClassicEnv.load(dirpath.joinpath("environment.json"))
        with open(dirpath.joinpath("physical_params.json"), 'r') as file:
            phy_param = json.load(file)
        # Instanciate the simulation
        simulation = cls(env.width, env.height, env.nb_obstacles, env._max_obs_size, env.pixel_size, env._seed, phy_param['diffusion'], phy_param['tmax'])
        simulation.reset()
        simulation.env = env
        # Set the parameter of the diffusion solver
        simulation.diffusion_solver.time = phy_param["time"]
        simulation.diffusion_solver.concentration = flow.field.read(str(dirpath.joinpath("concentration_field.npz")))
        return simulation

    def get_valid_point_cloud(self) -> flow.PointCloud:
        free_pos = np.array(self.env.free_positions, dtype=float) # array of free position in term of pixel size
        centers = self.env.pixel_size * (free_pos + 0.5)
        points = []
        for center in centers:
            points.append(flow.vec(x=center[0], y=center[1]))
        points = flow.tensor(points, flow.instance('point'))
        return flow.PointCloud(points, points, bounds=self.concentration.bounds)

    @staticmethod
    def get_gradient(field : flow.Grid) -> flow.CenteredGrid:
        flow.field.spatial_gradient(field)

    def start(self):
        if self.diffusion_solver is None:
            self.reset()
        self.diffusion_solver.start()

    @property
    def concentration(self) -> flow.CenteredGrid:
        "Concentration field at time Tmax in the given environment"
        if self.diffusion_solver is None:
            self.reset()
        return self.diffusion_solver.concentration

    @property
    def name(self) -> str:
        "Name used to identify the simulation"
        x_max, y_max = self.env.window_size
        return f"diffusion_Tmax={self.t_max}_D={self.diff}_Nobs={len(self.env.obstacles)}_size={self.env._max_obs_size}_Box({x_max},{y_max})_seed={self._seed}"

    def __repr__(self) -> str:
        return f"{__class__.__name__}(env={self.env!r}, solver={self.diffusion_solver!r})"
