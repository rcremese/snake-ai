import taichi as ti
import taichi.math as tm
import numpy as np

from snake_ai.envs import Rectangle
from snake_ai.taichi.field import VectorField

from abc import ABC, abstractmethod
from typing import List, Tuple, Union


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

        self._states = State.field(
            shape=(self.nb_walkers, 10),
        )
        assert (
            dt > 0.0 and diffusivity > 0.0
        ), f"Expected dt and diffusivity to be positive. Get {dt} and {diffusivity}"
        self.dt = dt
        self.diffusivity = diffusivity

    @ti.kernel
    def reset(self):
        for n in ti.ndrange(self.nb_walkers):
            self._states[n, 0].pos = self._init_pos[n]
            self._states[n, 0].vel = tm.vec2(0.0, 0.0)

    def collision_handling(self):
        return super().collision_handling()

    def compute_loss(self):
        return super().compute_loss()

    def optimize(self, target_pos: np.ndarray, max_iter: int = 1000, lr: float = 1e-3):
        return super().optimize(target_pos, max_iter, lr)

    @ti.kernel
    def step(self, t: int):
        for n in ti.ndrange(self.nb_walkers):
            self._states[n, t].pos = (
                self._states[n, t - 1].pos
                + self.dt * self._force_field._at_2d(self._states[n, t - 1].pos)
                + tm.sqrt(2 * self.dt * self.diffusivity)
                * tm.vec2(ti.randn(), ti.randn())
            )

    def run(self):
        return super().run()


if __name__ == "__main__":
    from pathlib import Path
    from snake_ai.utils.io import SimulationLoader
    from snake_ai.taichi.field import ScalarField, spatial_gradient
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
    force_field = spatial_gradient(concentration)
    starting_pos = np.array([[0.5, 0.5], [10.5, 10.5]])

    simulation = WalkerSimulationStoch2D(starting_pos, force_field, [], diffusivity=0.1)
    simulation.reset()
    simulation.step(1)
    simulation.step(2)

    print(simulation._states)
