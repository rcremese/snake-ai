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
    def __init__(self, init_pos : ArrayLike) -> None:
        self._init_pos = jnp.array(init_pos)
        self.reset()

    def reset(self):
        self.position = np.copy(self._init_pos)
        self.time = 0

    def step(self, gradient_field : GradientField):
        if not isinstance(gradient_field, GradientField):
            raise TypeError(f"Expected instance of GradientField, get {type(gradient_field)}")
        self.position += gradient_field(self.position)
        self.time += 1

    def draw(self, canvas : pygame.Surface):
        pygame.draw.circle(canvas, Colors.BLUE1.value, self.position, 1)
