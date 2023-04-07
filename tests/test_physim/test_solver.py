##
# @author  <robin.cremese@gmail.com>
# @file Test solvers
# @desc Created on 2023-04-07 2:34:32 pm
# @copyright MIT License
#
from snake_ai.physim.diffusion_solver import DiffusionSolver2D
from snake_ai.physim.gradient_field import compute_log
from snake_ai.envs import GridWorld
from pathlib import Path
from phi import flow
import numpy as np
import json
class TestDiffusionSolver:
    env = GridWorld()


from snake_ai.physim.simulation import Simulation
class TestSimulation:
    width, height = 10, 10
    pixel = 10
    nb_obs = 5
    t_max = 10
    diff = 1
    seed = 0


    def test_write_and_load(self, tmp_path : Path):
        simu = Simulation(self.width, self.height, self.nb_obs, self.pixel, self.seed, self.diff,  self.t_max)
        simu.start()
        simu.write(tmp_path)
        dirpath = tmp_path.joinpath(f"diffusion_Tmax={self.t_max}_D={self.diff}_Nobs={self.nb_obs}_Box({self.width * self.pixel},{self.height * self.pixel})_seed={self.seed}")
        assert dirpath.exists()
        assert dirpath.joinpath("environment.json").exists()
        assert dirpath.joinpath("concentration_field.npz").exists()
        assert dirpath.joinpath("physical_params.json").exists()
        with open(dirpath.joinpath("physical_params.json"), "r") as file:
            phy_param = json.load(file)
        assert phy_param == {"tmax" : self.t_max, "diffusion" : self.diff, "time" : self.t_max, "dt" : 1}
        # Test load
        new_simu = Simulation.load(dirpath)
        assert new_simu.concentration == simu.concentration
        assert new_simu.env == simu.env
