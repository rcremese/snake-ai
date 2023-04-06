from abc import abstractmethod, ABCMeta
from phi.jax import flow

class Solver(metaclass=ABCMeta):
    """Base class for solving the physics simulations PDE"""
    def __init__(self, initial_field : flow.CenteredGrid, obs_mask : flow.CenteredGrid, Tmax : float, dt : float, history_step : float = 0) -> None:
        assert isinstance(initial_field, flow.CenteredGrid), f"Expected initial field to be a phi.jax.flow.CenteredGrid, not {type(initial_field)}"
        self._initial_field = initial_field
        assert isinstance(obs_mask, flow.CenteredGrid), f"Expected obstacle mask to be a phi.jax.flow.CenteredGrid, not {type(obs_mask)}"
        assert obs_mask.shape == initial_field.shape, f"Expected obstacle mask and initial field to have the same shape, not {obs_mask.shape} and {initial_field.shape}"
        self._obs_mask = obs_mask
        assert dt > 0, f"Expected dt to be a positive float, not {dt}"
        self.dt = dt
        assert Tmax > 0, f"Expected Tmax to be a positive float, not {Tmax}"
        self.Tmax = Tmax
        assert 0 <= history_step <= Tmax, f"Expected history_step to be positive and less than Tmax, not {history_step} > {Tmax}."
        self._history_step = history_step
        # Initialize the field and time
        self.reset()

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def solve(self) -> None:
        raise NotImplementedError()
    
    def reset(self) -> None:
        self._field = self._initial_field
        self.time = 0

@flow.math.jit_compile
def explicit_diffusion(concentration : flow.field.Field, obstacle_mask : flow.field.Field, diffusivity : float, dt : float) -> flow.field.Field:
    return (1-obstacle_mask) * flow.diffuse.explicit(concentration, diffusivity=diffusivity, dt=dt)

@flow.math.jit_compile
def implicit_diffusion(concentration : flow.field.Field, obstacle_mask : flow.field.Field, diffusivity : float, dt : float) -> flow.field.Field:
    return (1-obstacle_mask) * flow.diffuse.implicit(concentration, diffusivity=diffusivity, dt=dt)

@flow.math.jit_compile
def crank_nicolson_diffusion(concentration : flow.field.Field, obstacle_mask : flow.field.Field, diffusivity : float, dt : float) -> flow.CenteredGrid:
    return (1-obstacle_mask) * flow.diffuse.implicit(concentration, diffusivity=diffusivity, dt=dt, order=2)

class DiffusionSolver(Solver):
    def __init__(self, initial_field: flow.CenteredGrid, obs_mask: flow.CenteredGrid, Tmax: float, dt: float, diffusivity : float, history_step : float = 0) -> None:
        assert diffusivity > 0, f"Expected diffusion coefficient to be a positive float, not {diffusivity}"
        self.diffusivity = diffusivity
        super().__init__(initial_field, obs_mask, Tmax, dt, history_step)
    # TODO : Attendre que le probleme de jit_compile soit resolu pour utiliser crank_nicolson_diffusion
    def step(self) -> None:
        # self._field = crank_nicolson_diffusion(self._field, self._obs_mask, self.diffusivity, self.dt)
        self._field = explicit_diffusion(self._field, self._obs_mask, self.diffusivity, self.dt)
        
    def solve(self) -> None:
        if self._history_step > 0:
            self._history = [self._initial_field]
            h_coef = 1

        while self.time < self.Tmax:
            self.step()
            self.time += self.dt
            if (self._history_step > 0) and (self.time >= h_coef * self._history_step):
                self._history.append(self._field)
                h_coef += 1
    
    @property
    def concentration(self) -> flow.CenteredGrid:
        "Returns the current concentration field"
        return self._field
    
    @property
    def history(self) -> list[flow.CenteredGrid]:
        "Returns the history of the concentration field"
        if self._history_step == 0:
            raise ValueError("The history is not recorded. Set history_step to a positive value to record the history.")
        return self._history

class EndlessDiffusionSolver(DiffusionSolver):
    def step(self) -> None:
        super().step()
        self._field += self._initial_field

if __name__ == "__main__":
    from snake_ai.envs import MazeGrid
    from snake_ai.physim.converter import DiffusionConverter, ObstacleConverter 
    import matplotlib.pyplot as plt

    env = MazeGrid()
    env.reset()
    converter = DiffusionConverter("pixel", 1)
    obs_converter = ObstacleConverter("pixel")
    initial_field = converter(env)
    obstacles = obs_converter(env)
    solver = EndlessDiffusionSolver(initial_field, obstacles, 100, 0.1, 1)
    solver.solve()
    flow.vis.plot(flow.math.log(solver.concentration))
    plt.show()

    
