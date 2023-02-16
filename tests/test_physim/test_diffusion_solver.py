from snake_ai.physim.diffusion_solver import DiffusionSolver2D
from snake_ai.physim.gradient_field import compute_log
from snake_ai.envs import SnakeClassicEnv
from pathlib import Path
from phi import flow
import numpy as np
import json
class TestDiffusionSolver:
    env = SnakeClassicEnv(nb_obstacles=10)
    env.seed()
    env.reset()
    x_max, y_max = env.window_size
    t_max = 1

    def test_init(self):
        solver = DiffusionSolver2D(self.x_max, self.y_max, self.t_max, source=self.env.food, obstacles=self.env.obstacles)
        assert solver._x_max == self.x_max
        assert solver._y_max == self.y_max
        assert solver.t_max == 1
        assert solver._grid_res == self.env.window_size
        assert solver.obstacles == [obs.to_phiflow() for obs in self.env.obstacles]
        solver_without_obs = DiffusionSolver2D(self.x_max, self.y_max, self.t_max, source=self.env.food)
        assert solver_without_obs.obstacles == []

    def test_log(self):
        field = flow.CenteredGrid(flow.math.random_uniform(flow.spatial(x=10, y=10)), bounds=flow.Box(x=1, y=1))
        initial_values = field.values.numpy('x,y')
        log_field = compute_log(field, eps=1e-5)
        assert np.isclose(log_field.values.numpy('x,y'), np.where(initial_values < 1e-5, np.log(1e-5), np.log(initial_values))).all()

    def test_solver(self):
        pass

    def test_step(self):
        pass

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
