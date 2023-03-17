##
# @author Robin CREMESE <robin.cremese@gmail.com>
# @file Class definition of the diffusion process
# @desc Created on 2022-11-16 5:29:26 pm
# @copyright https://mit-license.org/
#
from snake_ai.envs.snake_classic_env import SnakeClassicEnv
from snake_ai.utils import Colors
from snake_ai.utils.errors import ShapeError
from snake_ai.physim import Particle
from typing import Iterable, Tuple, Union, List, Optional, Dict
from functools import partial
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import logging
import pygame
import time
import jax


class DiffusionProcess():
    def __init__(self, nb_particles: float, t_max: int, window_size : Tuple[int], diff_coef: float = 1, part_radius: float = 1, obstacles : List[pygame.Rect] = None, seed: int = 0) -> None:
        if len(window_size) != 2:
            raise ValueError(f"The window size should be a 2-tuple of int representing the limits of the domain, got {window_size}")
        self.window_size = tuple([int(value) for value in window_size])

        if nb_particles < 1:
            raise ValueError(
                f"Diffusion process expect at least 1 particle, got {nb_particles}")
        self._nb_particles = int(nb_particles)

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

        if obstacles is None:
            obstacles = []
        self.obstacles = obstacles

        # seed everything
        self.seed(seed)

        # particles and collisions will be initialized when calling reset method
        self.time : int = None
        self._init_position : Tuple = None
        self._positions : jax.Array = None
        self._collisions : jax.Array = None
        self._concentration_map : jax.Array = None

    def reset(self, x_init : int, y_init : int) -> None:
        self._concentration_map = None
        self._init_particles(x_init, y_init)
        self.time = 0

    ## TODO : remove this chuck of data

    def start_simulation(self, draw=False):
        while self.time < self.t_max and np.sum(self._collisions) < self._nb_particles:
            self.step()
            if draw:
                self.draw()

    def step(self):
        if (self._collisions is None) or (self._positions is None):
            raise AttributeError(f"Collisions and positions not initalised. Call reset method to initialise at a position.")

        self._positions, self._key = self.update_positions(self._positions, self._collisions, self._key, self._diff_coef)
        self._collisions = self.check_collisions(self._positions, self.window_size, self._obstacles)
        self._concentration_map = None
        self.time += 1

    def draw(self):
        pygame.init()
        window = pygame.display.set_mode(self.window_size)

        canvas = pygame.Surface(self.window_size)
        canvas.fill(Colors.BLACK.value)
        # Draw snake, obstacles and food
        for obstacle in self.obstacles:
            pygame.draw.rect(canvas, color=Colors.RED.value, rect=obstacle)
        # Draw particles in the environment
        for particle in self.particles:
            particle.draw(canvas)
        # draw on the current display
        window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)

    ## Static methods
    @staticmethod
    @jax.jit
    def update_positions(positions : jax.Array, collisions : jax.Array, rng_key : int, diff_coef : float = 1) -> Tuple[jax.Array, int]:
        new_key, subkey = jax.random.split(rng_key)
        # update positions of points that do not collide with obstacles
        diffused_position = positions + jnp.sqrt(diff_coef) * jax.random.normal(subkey, shape=positions.shape)

        positions = jnp.where(collisions[:, None].repeat(2, axis=1), positions, diffused_position)
        return positions, new_key

    @staticmethod
    @jax.jit
    def check_collisions(positions : Union[jax.Array, np.ndarray], window_size : Tuple, obstacles : List[Dict[str, int]]) -> Union[jax.Array, np.ndarray]:
        assert positions.ndim == 2 and positions.shape[1] ==  2, f"Expected a 2D vector of shape (X, 2) where X is arbitrary. Get {positions.shape} instead."
        assert isinstance(obstacles, list) and all([obs.keys() == {'top', 'bottom', 'left', 'right'} for obs in obstacles])
        assert len(window_size) == 2

        width, height = window_size
        # Check collisions of a point outside the environment ( & : bitwise AND, | : bitwise OR)
        collisions = (positions[:, 0] < 0) | (positions[:, 0] > width) | (positions[:, 1] < 0) | (positions[:, 1] > height)
        # Check collisions of a point with obstacles
        for obstacle in obstacles:
            collisions |= (((positions[:, 0] >= obstacle['left']) & (positions[:, 0] <= obstacle['right'])) & \
                ((positions[:, 1] >= obstacle['top']) & (positions[:, 1] <= obstacle['bottom'])))
        return collisions

    @staticmethod
    @jax.jit
    def count_occurance(x : int, y : int, positions_int : jax.Array, collisions : jax.Array) -> int:
        return jnp.sum(~collisions & (positions_int[:,0] == x) & (positions_int[:,1] == y))

    # Properties
    @property
    def particles(self) -> List[Particle]:
        "List of particles"
        return [Particle(*pos, self._part_radius) for pos in self._positions.tolist()]

    @property
    def obstacles(self) -> List[pygame.Rect]:
        "List of obstacles as pygame rectangles"
        return [pygame.Rect(obs['left'], obs['top'], obs['right'] - obs['left'], obs['bottom'] - obs['top']) for obs in self._obstacles]

    @obstacles.setter
    def obstacles(self, obstacle_list : List[pygame.Rect]):
        if not isinstance(obstacle_list, list) and all([isinstance(obs, pygame.Rect) for obs in obstacle_list]):
            raise ValueError(f"Given obstacles are instance of {type(obstacle_list)}. Expected List[pygame.Rect]")
        self._obstacles = [{'left' : obs.left, 'right' : obs.right, 'top' : obs.top, 'bottom' : obs.bottom} for obs in obstacle_list]

    @property
    def nb_particles(self) -> int:
        "Number of particles in the environment"
        return self._nb_particles

    @nb_particles.setter
    def nb_particles(self, nb_part : float):
        "Setter method for nb_particles. If used reset positions and collision arrays"
        if nb_part < 1:
            raise ValueError(
                f"Diffusion process expect at least 1 particle, got {nb_part}")
        self._nb_particles = int(nb_part)
        self._init_particles(*self._init_position)

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
        self._collisions = self.check_collisions(self._positions, self.window_size, self._obstacles)
        self._concentration_map = None

    @property
    def concentration_map(self) -> jax.Array:
        "Particles concentration map as a JAX array"
        if self._concentration_map is None:
            if self.nb_particles < self.window_size[0] * self.window_size[1]:
                self._concentration_map = self._compute_concentration_map()
            else:
                self._concentration_map = self._accelerated_concentration_map()
        return self._concentration_map

    # Private methods
    def _init_particles(self, x : int, y : int):
        self._init_position = (x, y)
        self._positions = jnp.array([[x, y]]).repeat(self._nb_particles, axis=0)
        self._collisions = jnp.zeros(self._nb_particles, dtype=bool)

    def _compute_concentration_map(self) -> jax.Array:
        concentration_map = np.zeros(self.window_size)
        for x, y in self._positions[~self._collisions].astype(int):
            concentration_map[x, y] += 1
        return jnp.array(concentration_map)

    def _accelerated_concentration_map(self) -> jax.Array:
        positions_int = self._positions.astype(int)

        concentration_map = np.zeros(self.window_size, dtype=int)
        for x in jnp.arange(self.window_size[0]):
            for y in jnp.arange(self.window_size[1]):
                concentration_map[x, y] = self.count_occurance(x, y, positions_int, self._collisions)
        return concentration_map

    def __repr__(self) -> str:
        return f"{__class__.__name__}(nb_particles={self._nb_particles!r}, t_max={self.t_max!r}, diff_coef={self._diff_coef!r}, \
        part_radius={self._part_radius!r}, seed={self._key!r})"
