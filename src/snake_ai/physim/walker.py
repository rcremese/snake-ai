##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2023-01-12 2:04:03 pm
 # @copyright https://mit-license.org/
 #
import jax.numpy as jnp
import numpy as np

from snake_ai.utils.types import ArrayLike
from snake_ai.physim import GradientField
from snake_ai.utils import Colors

import pygame
class Walker:
    def __init__(self, init_pos : ArrayLike, dt : float = 1, sigma : float = 1) -> None:
        self._init_pos = jnp.array(init_pos)
        assert dt > 0
        self._dt = dt
        self._sigma = sigma
        self.reset()

    def reset(self):
        self.position = np.copy(self._init_pos)
        self.time = 0

    def step(self, gradient_field : GradientField):
        if not isinstance(gradient_field, GradientField):
            raise TypeError(f"Expected instance of GradientField, get {type(gradient_field)}")
        grad = gradient_field(self.position)
        print(grad)
        self.position += self._dt * grad + self._sigma * np.sqrt(self._dt) * np.random.normal(size=2)
        self.time += 1

    def draw(self, canvas : pygame.Surface):
        pygame.draw.circle(canvas, Colors.BLUE1.value, self.position.tolist(), 10)
