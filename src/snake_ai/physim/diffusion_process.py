##
# @author Robin CREMESE <robin.cremese@gmail.com>
# @file Class definition of the diffusion process
# @desc Created on 2022-11-16 5:29:26 pm
# @copyright https://mit-license.org/
#
from snake_ai.envs import SnakeClassicEnv
from snake_ai.utils import Colors, ShapeError
from snake_ai.physim import Particle
from typing import Iterable, Tuple, Union, List, Optional
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import logging
import pygame
import time
import jax


class DiffusionProcess():
    def __init__(self, env: SnakeClassicEnv, nb_particles: int, t_max: int, diff_coef: float = 1, part_radius: float = 1, conv_window : Optional[Tuple[int]] = None, seed: int = 0) -> None:
        # Check all the types of a class
        if not isinstance(env, SnakeClassicEnv):
            raise TypeError(
                "Diffusion process expect an environment of type SnakeBaseEnv.")
        self.env = env

        if nb_particles < 1:
            raise ValueError(
                f"Diffusion process expect at least 1 particle, got {nb_particles}")
        self._nb_particles = nb_particles

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

        if conv_window is None:
            # Gaussian convolutional window with the size of a pixel, centered with a scale 1
            conv_window = (self.env.pixel_size, 'gaussian', 1)
        self.conv_window = conv_window

        # seed everything
        self.seed(seed)

        # particles and collisions will be initialized when calling reset method
        self.time : int = None
        self._positions : jax.Array = None
        self._collisions : jax.Array = None
        self._concentration_map : jax.Array = None

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self._init_particles()
        self.time = 0
        return obs

    def start_simulation(self, draw=False):
        while self.time < self.t_max and np.sum(self._collisions) < self._nb_particles:
            self.step()
            if draw:
                self.draw()

    def step(self):
        self._key, subkey = jax.random.split(self._key)
        # update positions of points that do not collide with obstacles
        diffused_position = self._positions + jnp.sqrt(self._diff_coef) * jax.random.normal(subkey, shape=(self._nb_particles, 2)).block_until_ready()
        diffused_position.block_until_ready()

        self._positions = jnp.where(jnp.stack((self._collisions, self._collisions), axis=1), self._positions, diffused_position)
        self._positions.block_until_ready()
        self._collisions = self.check_collisions()
        self.time += 1

    def check_collisions(self, positions : Union[jax.Array, np.ndarray] = None) -> Union[jax.Array, np.ndarray]:
        if positions is None:
            positions = self._positions
        assert positions.ndim == 2 and positions.shape[1] ==  2, f"Expected a 2D vector of shape (X, 2) where X is arbitrary. Get {positions.shape} instead."

        env_width, env_height = self.env.window_size
        # Check collisions of a point outside the environment ( & : bitwise AND, | : bitwise OR)
        collisions = (positions[:, 0] < 0) | (positions[:, 0] > env_width) | (positions[:, 1] < 0) | (positions[:, 1] > env_height)
        # Check collisions of a point with obstacles
        for obstacle in self.env.obstacles:
            collisions |= (((positions[:, 0] >= obstacle.left) & (positions[:, 0] <= obstacle.right)) & \
                ((positions[:, 1] >= obstacle.top) & (positions[:, 1] <= obstacle.bottom)))
            if isinstance(positions, jax.Array):
                collisions.block_until_ready()
        return collisions

    def set_source_position(self, x: int, y: int):
        self.env.food = pygame.Rect(
            x, y, self.env.food.width, self.env.food.height)
        self._init_particles()
        self.env.check_overlaps()

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
    def particles(self) -> List[Particle]:
        "List of particles"
        return [Particle(*pos, self._part_radius) for pos in self._positions.tolist()]

    @property
    def nb_particles(self) -> int:
        "Number of particles in the environment"
        return self._nb_particles

    @nb_particles.setter
    def nb_particles(self, nb_part : int):
        "Setter method for nb_particles. If used reset positions and collision arrays"
        if nb_part < 1:
            raise ValueError(
                f"Diffusion process expect at least 1 particle, got {nb_part}")
        self._nb_particles = nb_part
        self._init_particles()

    @property
    def positions(self) -> jax.Array:
        "Positions of the particles as a JAX array"
        return self._positions

    @positions.setter
    def positions(self, positions : Union[np.ndarray, jax.Array]):
        if not isinstance(positions, (np.ndarray, jax.Array)):
            raise TypeError(f"Expected instance of np.ndarray, jax.Array. Get {type(positions)} instead.")
        if positions.shape != (self._nb_particles, 2):
            raise ShapeError(f"The expected size is ({self._nb_particles}, 2). Get {positions.shape} instead.")
        self._positions = jnp.array(positions)
        self._collisions = self.check_collisions()

    @property
    def concentration_map(self) -> jax.Array:
        "Particles concentration map as a JAX array"
        if self._concentration_map is None:
            self._compute_concentration_map()
        return self._concentration_map

    # Private methods
    def _init_particles(self):
        self._positions = jnp.repeat(jnp.array(
            [self.env.food.center], dtype=float), repeats=self._nb_particles, axis=0)
        self._collisions = jnp.zeros(self._nb_particles, dtype=bool)

    def _compute_concentration_map(self):
        concentration_map = np.zeros(self.env.window_size, dtype=int)
        for x, y in self._positions[~self._collisions]:
            concentration_map[int(x), int(y)] += 1
        self._concentration_map = jnp.array(concentration_map)

    def __repr__(self) -> str:
        return f"{__class__.__name__}({self.env!r}, nb_particles={self._nb_particles!r}, t_max={self.t_max!r}, diff_coef={self._diff_coef!r}, \
        part_radius={self._part_radius!r}, conv_window={self.conv_window!r}, seed={self._key!r})"
