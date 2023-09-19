import taichi as ti
import taichi.math as tm
import numpy as np

from snake_ai.envs import Rectangle
from snake_ai.taichi.field import ScalarField, VectorField

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union


@ti.dataclass
class State:
    pos: tm.vec2
    vel: tm.vec2


@ti.data_oriented
class WalkerSimulation2D(ABC):
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self):
        raise NotImplementedError

    @abstractmethod
    def collision_handling(self):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self):
        raise NotImplementedError

    @abstractmethod
    def optimize(self, target_pos: np.ndarray, max_iter: int = 1000, lr: float = 1e-3):
        raise NotImplementedError


@ti.data_oriented
class WalkerSimulationStoch2D(WalkerSimulation2D):
    def __init__(
        self,
        positions: np.ndarray,
        force_field: VectorField,
        obstacles: List[Rectangle],
        t_max: int = 100,
        dt: float = 0.1,
        diffusivity: float = 1.0,
    ):
        assert (
            positions.ndim == 2 and positions.shape[1] == 2
        ), "Expected position to be a (n, 2)-array of position vectors. Get {}".format(
            positions.shape
        )
        self.nb_walkers = positions.shape[0]
        self._init_pos = ti.Vector.field(2, dtype=ti.f32, shape=(self.nb_walkers,))
        self._init_pos.from_numpy(positions)

        assert isinstance(
            force_field, VectorField
        ), "Expected force_field to be a VectorField. Get {}".format(type(force_field))
        self._force_field = force_field

        self._states = State.field(shape=(self.nb_walkers, t_max), needs_grad=True)
        self._noise = ti.Vector.field(
            2, dtype=float, shape=(self.nb_walkers, t_max), needs_grad=True
        )
        assert (
            dt > 0.0 and diffusivity >= 0.0
        ), f"Expected dt and diffusivity to be positive. Get {dt} and {diffusivity}"
        self.dt = dt
        self.nb_steps = t_max
        self.diffusivity = diffusivity
        # Definition of the loss
        self.loss = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def reset(self):
        for n in ti.ndrange(self.nb_walkers):
            self._states[n, 0].pos = self._init_pos[n]
            self._states[n, 0].vel = tm.vec2(0.0, 0.0)
            for t in range(self.nb_steps):
                self._noise[n, t] = tm.vec2(ti.randn(), ti.randn())

    def collision_handling(self):
        return super().collision_handling()

    @ti.kernel
    def compute_loss(self, t: int):
        for n in range(self.nb_walkers):
            self.loss[None] += (
                tm.length(self._states[n, t].pos - self.target)
            ) / self.nb_walkers

    def optimize(self, target_pos: np.ndarray, max_iter: int = 100, lr: float = 1e-3):
        self.target = tm.vec2(*target_pos)
        for iter in range(max_iter):
            self.reset()
            with ti.ad.Tape(self.loss):
                self.run()
                self.compute_loss(self.nb_steps - 1)
            print("Iter=", iter, "Loss=", self.loss[None])
            self._update_force_field(lr)

    @ti.kernel
    def _update_force_field(self, lr: float):
        for i, j in self._force_field._values:
            self._force_field._values[i, j] -= lr * self._force_field._values.grad[i, j]

    @ti.kernel
    def step(self, t: int):
        for n in ti.ndrange(self.nb_walkers):
            self._states[n, t].pos = (
                self._states[n, t - 1].pos
                + self.dt * self._force_field._at_2d(self._states[n, t - 1].pos)
                + tm.sqrt(2 * self.dt * self.diffusivity) * self._noise[n, t]
            )

    def run(self):
        for t in range(1, self.nb_steps):
            self.step(t)

    @property
    def trajectories(self) -> Dict[str, np.ndarray]:
        return self._states.to_numpy()


def render(
    simulation: WalkerSimulationStoch2D,
    concentration: ScalarField,
    window_size: Tuple[int, int],
):
    gui = ti.GUI("Differentiable Simulation", window_size)
    scale_vector = np.array(
        concentration._bounds.width(), concentration._bounds.height()
    )
    trajectories = simulation.trajectories
    for t in range(simulation.nb_steps):
        gui.contour(concentration._values, normalize=True)
        pos = trajectories["pos"][:, t] / scale_vector
        gui.circles(pos, radius=5, color=int(ti.rgb_to_hex([255, 0, 0])))
        gui.show()


if __name__ == "__main__":
    from pathlib import Path
    from snake_ai.utils.io import SimulationLoader
    from snake_ai.taichi.field import ScalarField, spatial_gradient, log
    from snake_ai.taichi.geometry import Box2D

    ti.init(debug=True)
    dirpath = Path("/home/rcremese/projects/snake-ai/simulations").resolve(strict=True)
    filepath = dirpath.joinpath(
        "Slot(20,20)_pixel_Tmax=800.0_D=1", "seed_10", "field.npz"
    )
    obj = np.load(filepath)

    ti.init(debug=True)
    bounds = Box2D(obj["lower"], obj["upper"])
    concentration = ScalarField(obj["data"], bounds)
    log_concentration = log(concentration)
    force_field = spatial_gradient(log_concentration, needs_grad=True)
    starting_pos = np.array([[0.5, 0.5], [10.5, 10.5], [0.5, 10.5], [15.5, 10.5]])

    simulation = WalkerSimulationStoch2D(
        starting_pos, force_field, [], t_max=100, diffusivity=0
    )
    simulation.reset()
    simulation.run()
    print(simulation.trajectories["pos"][:, -1])
    render(simulation, concentration, (500, 500))
    simulation.optimize(np.array([10.5, 5.0]), max_iter=100, lr=1)
    render(simulation, concentration, (500, 500))
    # simulation.optimize(np.array([13.5, 13.5]), lr=0.1)
