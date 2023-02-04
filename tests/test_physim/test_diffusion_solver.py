from snake_ai.physim.diffusion_solver import DiffusionSolver2D
from snake_ai.envs import SnakeClassicEnv

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

    def test_rest(self):
        pass

    def test_solver(self):
        pass

    def test_step(self):
        pass
