from snake_ai.envs import SnakeClassicEnv
from snake_ai.envs.snake_base_env import SnakeBaseEnv
from snake_ai.physim import Particle
import numpy as np

class DiffusionProcess():
    def __init__(self, env : SnakeClassicEnv, nb_particles : int, t_max : int, diff_coef : float = 1, part_radius : float = 1) -> None:
        # Check all the types of a class
        if not isinstance(env, SnakeBaseEnv):
            raise TypeError("Diffusion process expect an environment of type SnakeBaseEnv.")
        self.env = env

        if nb_particles < 1:
            raise ValueError(f"Diffusion process expect at least 1 particle, got {nb_particles}")
        self.nb_particles = nb_particles

        if t_max < 1:
            raise ValueError(f"Diffusion process expect the maximum time to be at least 1 time step, got {diff_coef}")
        self.t_max = t_max

        if diff_coef <= 0:
            raise ValueError(f"Diffusion process expect the diffusion coefficient to be positive, got {diff_coef}")
        self._diff_coef = diff_coef

        if part_radius <= 0:
            raise ValueError(f"Diffusion process expect the particles radius to be positive, got {part_radius}")
        self._part_radius = part_radius
        # particles and collisions will be initialized when calling reset method
        self.particles = None
        self._positions = None
        self._collisions = None
        self._nb_collision_free = None

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self.particles = [Particle(*self.env.food.center, self._part_radius) for _ in range(self.nb_particles)]
        self._positions = np.repeat([self.env.food.center], repeats=self.nb_particles, axis=0)
        self._collisions = np.zeros(self.nb_particles, dtype=bool)
        self._nb_collision_free = self.nb_particles
        return obs

    def start(self):
        t = 0
        while t < self.t_max and self._nb_collision_free < self.nb_particles :
            self.step()
            t += 1

    def step(self):
        self._positions[~self._collisions] += self._diff_coef * np.random.randn(2, self._nb_collision_free)
        for idx, position in enumerate(self._positions):
            if self._collisions[idx]:
                continue
            self.particles[idx].update_position(*position)
            self._collisions[idx] = self.particles[idx].collide_any(self.env.obstacles)
        self._nb_collision_free = np.sum(~self._collisions)

    def __repr__(self) -> str:
        return f"{__class__.__name__}({self.env!r}, {self.nb_particles!r}, {self.t_max!r},{self._diff_coef!r}, {self._part_radius!r})"

