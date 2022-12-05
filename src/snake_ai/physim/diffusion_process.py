##
# @author Robin CREMESE <robin.cremese@gmail.com>
# @file Class definition of the diffusion process
# @desc Created on 2022-11-16 5:29:26 pm
# @copyright https://mit-license.org/
#
from snake_ai.envs import SnakeClassicEnv
from snake_ai.envs.snake_base_env import SnakeBaseEnv
from snake_ai.utils import Colors
from snake_ai.physim import Particle
import jax.numpy as jnp
import numpy as np
import pygame
import time
import jax


class DiffusionProcess():
    def __init__(self, env: SnakeClassicEnv, nb_particles: int, t_max: int, diff_coef: float = 1, part_radius: float = 1, seed: int = 0) -> None:
        # Check all the types of a class
        if not isinstance(env, SnakeBaseEnv):
            raise TypeError(
                "Diffusion process expect an environment of type SnakeBaseEnv.")
        self.env = env

        if nb_particles < 1:
            raise ValueError(
                f"Diffusion process expect at least 1 particle, got {nb_particles}")
        self.nb_particles = nb_particles

        if t_max < 1:
            raise ValueError(
                f"Diffusion process expect the maximum time to be at least 1 time step, got {diff_coef}")
        self.t_max = t_max

        if diff_coef <= 0:
            raise ValueError(
                f"Diffusion process expect the diffusion coefficient to be positive, got {diff_coef}")
        self._diff_coef = diff_coef

        if part_radius <= 0:
            raise ValueError(
                f"Diffusion process expect the particles radius to be positive, got {part_radius}")
        self._part_radius = part_radius

        # seed everything
        self.seed(seed)

        # particles and collisions will be initialized when calling reset method
        self.time = None
        self._positions = None
        self._collisions = None

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self._init_particles()
        self.time = 0
        return obs

    def start_simulation(self):
        t = 0
        while t < self.t_max and np.sum(self._collisions) < self.nb_particles:
            self.step()
            self.draw()
            t += 1

    def step(self):
        self._key, subkey = jax.random.split(self._key)

        # update positions of points that do not collide with obstacles
        diffused_position = self._positions + jnp.sqrt(self._diff_coef) * jax.random.normal(subkey, shape=(self.nb_particles, 2)).block_until_ready()
        diffused_position.block_until_ready()

        self._positions = jnp.where(jnp.stack((self._collisions, self._collisions), axis=1), self._positions, diffused_position)
        self._positions.block_until_ready()
        self._collisions = self.check_collisions(self._positions)
        self.time += 1

    def check_collisions(self, positions : jax.Array) -> jax.Array:
        env_width, env_height = self.env.window_size

        # Check collisions of a point outside the environment ( & : bitwise AND, | : bitwise OR)
        collisions = (positions[:, 0] < 0) | (positions[:, 0] > env_width) | (positions[:, 1] < 0) | (positions[:, 1] > env_height)
        # Check collisions of a point with obstacles 
        for obstacle in self.env.obstacles:
            collisions |= (((positions[:, 0] >= obstacle.left) & (positions[:, 0] <= obstacle.right)) & \
                ((positions[:, 1] >= obstacle.top) & (positions[:, 1] <= obstacle.bottom)))
            collisions.block_until_ready()
        return collisions

        # for idx, position in enumerate(self._positions):
        #     if self._collisions[idx]:
        #         continue
        #     self.particles[idx].update_position(*position)
        #     self._collisions[idx] = self.particles[idx].collide_any(self.env.obstacles) or (not self.particles[idx].is_inside(playground))

    def set_source_position(self, x: int, y: int):
        self.env.food = pygame.Rect(
            x, y, self.env.food.width, self.env.food.height)
        # TODO : implement self.env.check_collisions()
        self._init_particles()

    def draw(self):
        pygame.init()
        window = pygame.display.set_mode(self.env.window_size)

        canvas = pygame.Surface(self.env.window_size)
        canvas.fill(Colors.BLACK.value)
        # Draw snake, obstacles and food
        self.env.draw(canvas)
        # Draw particles in the environment
        for particle in self.particles:
            particle.draw(canvas)
        # draw on the current display
        window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

    def seed(self, seed: int = 0):
        self.env.seed(seed)
        self._key = jax.random.PRNGKey(seed)

    # Properties
    @property
    def particles(self):
        "List of particles"
        return [Particle(*pos, self._part_radius) for pos in self._positions.tolist()]

    # Private methods
    def _init_particles(self):
        self._positions = jnp.repeat(jnp.array(
            [self.env.food.center], dtype=float), repeats=self.nb_particles, axis=0)
        self._collisions = jnp.zeros(self.nb_particles, dtype=bool)

    def __repr__(self) -> str:
        return f"{__class__.__name__}({self.env!r}, {self.nb_particles!r}, {self.t_max!r},{self._diff_coef!r}, {self._part_radius!r})"


if __name__ == '__main__':
    env = SnakeClassicEnv(nb_obstacles=10)
    diff_process = DiffusionProcess(
        env, nb_particles=1_000, t_max=100, diff_coef=100, part_radius=2)
    diff_process.reset()
    diff_process.draw()
    time.sleep(5)
    diff_process.start_simulation()
    diff_process.draw()
    print(diff_process.time)
    time.sleep(5)
