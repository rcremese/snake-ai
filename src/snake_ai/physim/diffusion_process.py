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
import numpy as np
import pygame
import time

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
        while t < self.t_max and np.sum(self._collisions) < self.nb_particles :
            self.step()
            self.draw()
            t += 1

    def step(self):
        # update positions of points that do not collide with obstacles
        playground = pygame.Rect(0, 0, *self.env.window_size)

        self._positions[~self._collisions] += self._diff_coef * np.random.randn(self.nb_particles, 2)[~self._collisions]
        for idx, position in enumerate(self._positions):
            if self._collisions[idx]:
                continue
            self.particles[idx].update_position(*position)
            self._collisions[idx] = self.particles[idx].collide_any(self.env.obstacles) or (not self.particles[idx].is_inside(playground))
        self.time += 1

    def set_source_position(self, x : int, y : int):
        self.env.food = pygame.Rect(x, y, self.env.food.width, self.env.food.height)
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


    def seed(self, seed : int = 0):
        self.env.seed(seed)
        np.random.seed(seed)

    ## Private methods
    def _init_particles(self):
        self.particles = [Particle(*self.env.food.center, self._part_radius) for _ in range(self.nb_particles)]
        self._positions = np.repeat(np.array([self.env.food.center], dtype=float), repeats=self.nb_particles, axis=0)
        self._collisions = np.zeros(self.nb_particles, dtype=bool)

    def __repr__(self) -> str:
        return f"{__class__.__name__}({self.env!r}, {self.nb_particles!r}, {self.t_max!r},{self._diff_coef!r}, {self._part_radius!r})"

if __name__ == '__main__':
    env = SnakeClassicEnv(nb_obstacles=20)
    diff_process = DiffusionProcess(env, nb_particles=1_000, t_max=1_000, diff_coef=10, part_radius=2)
    diff_process.reset()
    diff_process.draw()
    time.sleep(10)
    diff_process.start_simulation()
    diff_process.draw()
    print(diff_process.time)

