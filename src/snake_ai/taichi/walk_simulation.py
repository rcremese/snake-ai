import taichi as ti
import taichi.math as tm
import numpy as np

from snake_ai.envs import Rectangle
from snake_ai.taichi.field import ScalarField, VectorField, spatial_gradient
from snake_ai.taichi.geometry import Box2D, convert_rectangles
from snake_ai.taichi.maths import lerp

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
    def collision_handling(self, n: int, t: int):
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
        potential_field: ScalarField,
        obstacles: List[Rectangle] = None,
        t_max: int = 100,
        dt: float = 0.1,
        diffusivity: float = 1.0,
    ):
        ## Initialisation of the walkers positions
        assert (
            positions.ndim == 2 and positions.shape[1] == 2
        ), "Expected position to be a (n, 2)-array of position vectors. Get {}".format(
            positions.shape
        )
        self.nb_walkers = positions.shape[0]
        self._init_pos = ti.Vector.field(2, dtype=ti.f32, shape=(self.nb_walkers,))
        self._init_pos.from_numpy(positions)

        ## Initialisation of the force field
        assert isinstance(
            potential_field, ScalarField
        ), "Expected force_field to be a VectorField. Get {}".format(
            type(potential_field)
        )
        self.force_field = spatial_gradient(potential_field, needs_grad=True)

        ## Initialisation of the obstacles
        if obstacles is None:
            obstacles = [Rectangle(0, 0, 0, 0)]  # Dummy obstacle
        assert isinstance(obstacles, (list, tuple)) and all(
            isinstance(obs, Rectangle) for obs in obstacles
        ), f"Expected obstacles to be a list of Rectangle. Get {obstacles}"
        self.nb_obstacles = len(obstacles)
        self.obstacles = convert_rectangles(obstacles)

        ## Simulation specific parameters
        self.states = State.field(shape=(self.nb_walkers, t_max), needs_grad=True)
        self._noise = ti.Vector.field(
            2, dtype=float, shape=(self.nb_walkers, t_max), needs_grad=False
        )
        assert (
            dt > 0.0 and diffusivity >= 0.0
        ), f"Expected dt and diffusivity to be positive. Get {dt} and {diffusivity}"
        self.dt = dt
        self.nb_steps = t_max
        self.diffusivity = diffusivity
        # Definition of the loss
        self.loss = ti.field(ti.f32, shape=(), needs_grad=True)

    ## Public methods
    def run(self):
        for t in range(1, self.nb_steps):
            self.step(t)

    def optimize(self, target_pos: np.ndarray, max_iter: int = 100, lr: float = 1e-3):
        assert isinstance(target_pos, np.ndarray) and target_pos.shape == (
            2,
        ), f"Expected target_pos to be a 2D-array. Get {target_pos.shape}"
        assert (
            isinstance(max_iter, int) and max_iter > 0
        ), f"Expected max_iter to be a positive integer. Get {max_iter}"
        assert (
            isinstance(lr, (float, int)) and lr > 0.0
        ), f"Expected lr to be a positive float. Get {lr}"

        self.target = tm.vec2(*target_pos)
        for iter in range(max_iter):
            self.reset()
            with ti.ad.Tape(self.loss):
                self.run()
                self.compute_loss(self.nb_steps - 1)
            print("Iter=", iter, "Loss=", self.loss[None])
            self._update_force_field(lr)

    ### Taichi kernels
    @ti.kernel
    def reset(self):
        for n in ti.ndrange(self.nb_walkers):
            self.states[n, 0].pos = self._init_pos[n]
            self.states[n, 0].vel = tm.vec2(0.0, 0.0)
            for t in ti.ndrange(self.nb_steps):
                self._noise[n, t] = tm.vec2(ti.randn(), ti.randn())

    @ti.kernel
    def step(self, t: int):
        for n in ti.ndrange(self.nb_walkers):
            self.states[n, t].pos = (
                self.states[n, t - 1].pos
                + self.dt * self.force_field._at_2d(self.states[n, t - 1].pos)
                + tm.sqrt(2 * self.dt * self.diffusivity) * self._noise[n, t]
            )
            self.collision_handling(n, t)

    @ti.func
    def collision_handling(self, n: int, t: int):
        """Check all collision for a walker n at time t and apply changes.

        Args:
            n (int): walker index
            t (int): timestep index
        """
        ## Domain collisions
        if not self.force_field.contains(self.states[n, t].pos):
            self._domain_collision(n, t)
        ## Obstacle collisions
        for o in ti.ndrange(self.nb_obstacles):
            if self.obstacles[o].contains(self.states[n, t].pos):
                self._obstacle_collision(n, t, o)

    @ti.kernel
    def compute_loss(self, t: int):
        for n in range(self.nb_walkers):
            self.loss[None] += (
                tm.length(self.states[n, t].pos - self.target)
            ) / self.nb_walkers

    ## Private methods
    @ti.func
    def _domain_collision(self, n: int, t: int):
        """Check the collision of a walker at time t with the domain boundaries.

        Args:
            n (int): index of the walker
            t (int): index of the time step
        """
        for i in ti.static(range(2)):
            if self.states[n, t].pos[i] < self.force_field.bounds.min[i]:
                self.states[n, t].pos[i] = self.force_field.bounds.min[i]
                self.states[n, t].vel[i] *= -1.0
            elif self.states[n, t].pos[i] > self.force_field.bounds.max[i]:
                self.states[n, t].pos[i] = self.force_field.bounds.max[i]
                self.states[n, t].vel[i] *= -1.0

    @ti.func
    def _obstacle_collision(self, n: int, t: int, o: int):
        for i in ti.static(range(2)):
            ## Collision on the min border
            if (self.states[n, t - 1].pos[i] < self.obstacles[o].min[i]) and (
                self.states[n, t].pos[i] >= self.obstacles[o].min[i]
            ):
                self.states[n, t].pos[i] = self.obstacles[o].min[i] - ti.abs(
                    self.obstacles[o].min[i] - self.states[n, t].pos[i]
                )
                self.states[n, t].vel[i] = -self.states[n, t].vel[i]
            # Collision on the max border
            elif (self.states[n, t - 1].pos[i] > self.obstacles[o].max[i]) and (
                self.states[n, t].pos[i] <= self.obstacles[o].max[i]
            ):
                self.states[n, t].pos[i] = self.obstacles[o].max[i] + ti.abs(
                    self.obstacles[o].max[i] - self.states[n, t].pos[i]
                )
                self.states[n, t].vel[i] = -self.states[n, t].vel[i]

    @ti.kernel
    def _update_force_field(self, lr: float):
        for i, j in self.force_field.values:
            self.force_field.values[i, j] -= lr * self.force_field.values.grad[i, j]

    ## Properties
    @property
    def trajectories(self) -> Dict[str, np.ndarray]:
        return self.states.to_numpy()

    @property
    def positions(self) -> np.ndarray:
        return self.states.pos.to_numpy()


## Dynamic case
# self.states[n, t].vel = (
#     self.states[n, t - 1].vel
#     + self.dt * self._force_field._at_2d(self.states[n, t - 1].pos)
#     + tm.sqrt(2 * self.dt * self.diffusivity) * self._noise[n, t]
# )
# self.states[n, t].pos = self.states[n, t - 1].pos + self.dt * self.states[n,t].vel


def render(
    simulation: WalkerSimulationStoch2D,
    concentration: ScalarField,
    window_size: Tuple[int, int],
):
    gui = ti.GUI("Differentiable Simulation", window_size)
    scale_vector = np.array(concentration.bounds.width(), concentration.bounds.height())
    trajectories = simulation.trajectories
    obstacles = simulation.obstacles.to_numpy()

    for t in range(simulation.nb_steps):
        gui.contour(concentration.values, normalize=True)
        pos = trajectories["pos"][:, t] / scale_vector
        gui.circles(pos, radius=5, color=int(ti.rgb_to_hex([255, 0, 0])))
        for obs_min, obs_max in zip(obstacles["min"], obstacles["max"]):
            gui.rect(
                topleft=obs_min / scale_vector,
                bottomright=obs_max / scale_vector,
                color=0xEEEEF0,
            )
        gui.show()


if __name__ == "__main__":
    from pathlib import Path
    from snake_ai.taichi.field import ScalarField, spatial_gradient, log
    from snake_ai.taichi.geometry import Box2D

    import snake_ai.utils.visualization as vis

    ti.init(arch=ti.gpu)
    dirpath = Path("/home/rcremese/projects/snake-ai/simulations").resolve(strict=True)
    filepath = dirpath.joinpath(
        "Slot(20,20)_pixel_Tmax=800.0_D=1", "seed_10", "field.npz"
    )
    obj = np.load(filepath)

    bounds = Box2D(obj["lower"], obj["upper"])
    concentration = ScalarField(obj["data"], bounds)
    log_concentration = log(concentration)
    # force_field = spatial_gradient(log_concentration, needs_grad=True)
    starting_pos = np.array([[0.5, 0.5], [10.5, 10.5], [0.5, 10.5], [15.5, 10.5]])

    simulation = WalkerSimulationStoch2D(
        starting_pos,
        log_concentration,
        # obstacles=[Rectangle(0, 0, 1, 1)],
        t_max=100,
        diffusivity=0,
    )
    simulation.reset()
    simulation.run()

    render(simulation, concentration, (500, 500))
    vis.animate_walk_history(
        simulation.positions,
        log_concentration.values.to_numpy(),
        bound_limits=obj["upper"],
        output_path="test.gif",
    )

    # simulation.optimize(np.array([10.5, 5.0]), max_iter=100, lr=1)
    # render(simulation, concentration, (500, 500))
