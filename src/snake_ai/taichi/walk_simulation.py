import taichi as ti
import taichi.math as tm
import numpy as np

from snake_ai.envs import Rectangle

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
class WalkerSimulationStoch(WalkerSimulation2D):
    def __init__(
        self,
        pos: np.ndarray,
        force_field: np.ndarray,
        obstacles: List[Rectangle],
    ):
        assert (
            pos.ndim == 2 and pos.shape[1] == 2
        ), "Expected position to be a (n, 2)-array of position vectors. Get {}".format(
            pos.shape
        )
        self._init_pos = pos


@ti.data_oriented
class WalkerSimulationStoch(WalkerSimulation2D):
    pass
