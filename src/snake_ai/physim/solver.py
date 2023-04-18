from abc import abstractmethod, ABCMeta
from phi.jax import flow
from typing import Union, List
from snake_ai.utils import errors

@flow.math.jit_compile
def explicit_diffusion(concentration : flow.field.Field, obstacle_mask : flow.field.Field, diffusivity : float, dt : float) -> flow.field.Field:
    return (1-obstacle_mask) * flow.diffuse.explicit(concentration, diffusivity=diffusivity, dt=dt, substeps=2)

@flow.math.jit_compile
def implicit_diffusion(concentration : flow.field.Field, obstacle_mask : flow.field.Field, diffusivity : float, dt : float) -> flow.field.Field:
    return (1-obstacle_mask) * flow.diffuse.implicit(concentration, diffusivity=diffusivity, dt=dt)

@flow.math.jit_compile
def crank_nicolson_diffusion(concentration : flow.field.Field, obstacle_mask : flow.field.Field, diffusivity : float, dt : float) -> flow.CenteredGrid:
    return (1-obstacle_mask) * flow.diffuse.implicit(concentration, diffusivity=diffusivity, dt=dt, order=2)

class DiffusionSolver:
    names = ["explicit", "implicit", "crank_nicolson"]
    def __init__(self, diffusivity : float, t_max: float, dt: float, history_step : float = 0, name : str = "crank_nicolson", endless : bool = False) -> None:
        if dt <= 0 or t_max <= 0 or diffusivity <= 0:
            raise ValueError(f"Expected diffusivity, Tmax and dt and to be positive floats, not {diffusivity}, {t_max} and {dt}")
        self.dt, self.t_max = dt, t_max
        self.diffusivity = diffusivity

        if history_step < 0 or history_step > t_max:
            raise ValueError(f"Expected history_step to be positive and less than Tmax, not {history_step} > {t_max}.")
        self._hist_step = history_step
        self._history = []

        if name.lower() not in DiffusionSolver.names:
            raise ValueError( f"Expected name to be one of {DiffusionSolver.names}, not {name}")
        self.name = name.lower()
        assert isinstance(endless, bool), f"Expected endless to be a boolean, not {type(endless)}"
        self._endless = endless

    # TODO : Attendre que le probleme de jit_compile soit resolu pour utiliser crank_nicolson_diffusion
    def step(self, field : flow.CenteredGrid, obs_mask : flow.CenteredGrid) -> flow.CenteredGrid:
        assert isinstance(field, flow.CenteredGrid) and isinstance(obs_mask, flow.CenteredGrid), \
            f"Expected initial field and obs_mask to be a phi.jax.flow.CenteredGrid, not {type(field)} and {type(obs_mask)}"
        assert obs_mask.shape == field.shape, f"Expected obstacle mask and initial field to have the same shape, not {obs_mask.shape} and {field.shape}"

        if self.name == "explicit":
            return explicit_diffusion(field, obs_mask, self.diffusivity, self.dt)
        elif self.name == "implicit":
            return implicit_diffusion(field, obs_mask, self.diffusivity, self.dt)
        elif self.name == "crank_nicolson":
            return crank_nicolson_diffusion(field, obs_mask, self.diffusivity, self.dt)

    def solve(self, initial_field : flow.CenteredGrid, obs_mask : flow.CenteredGrid) -> flow.CenteredGrid:
        assert isinstance(initial_field, flow.CenteredGrid) and isinstance(obs_mask, flow.CenteredGrid), \
            f"Expected initial field and obs_mask to be a phi.jax.flow.CenteredGrid, not {type(initial_field)} and {type(obs_mask)}"
        assert obs_mask.shape == initial_field.shape, f"Expected obstacle mask and initial field to have the same shape, not {obs_mask.shape} and {initial_field.shape}"

        if self._hist_step > 0:
            self._history = [initial_field]
            h_coef = 1

        field = initial_field
        time = 0
        while time < self.t_max:
            field = self.step(field, obs_mask)
            if self._endless:
                field += initial_field
            time += self.dt
            # Record the history
            if (self._hist_step > 0) and (time >= h_coef * self._hist_step):
                self._history.append(field)
                h_coef += 1
        return field

    @property
    def history(self) -> list[flow.CenteredGrid]:
        "Returns the history of the concentration field"
        if self._hist_step == 0:
            raise ValueError("The history is not recorded. Set history_step to a positive value to record the history.")
        return self._history

    @property
    def is_stationary(self) -> bool:
        return self._endless

    def __repr__(self) -> str:
        return f"{__class__.__name__}(diffusivity={self.diffusivity}, Tmax={self.t_max}, dt={self.dt}, history_step={self._hist_step}, solver={self.name}, endless={self._endless})"

if __name__ == "__main__":
    from snake_ai.envs import MazeGrid
    from snake_ai.physim.converter import DiffusionConverter, ObstacleConverter
    import matplotlib.pyplot as plt

    env = MazeGrid()
    env.reset()
    converter = DiffusionConverter("pixel")
    obs_converter = ObstacleConverter("pixel")
    field = converter(env)
    obstacles = obs_converter(env)
    solver = DiffusionSolver(1, 100, 0.1, endless=True)
    concentration = solver.solve(field, obstacles)
    flow.vis.plot(flow.math.log(concentration))
    plt.show()


