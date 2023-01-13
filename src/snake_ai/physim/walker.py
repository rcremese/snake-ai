##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2023-01-12 2:04:03 pm
 # @copyright https://mit-license.org/
 #
import jax.numpy as jnp
from snake_ai.utils.types import ArrayLike
from snake_ai.physim import GradientField

class Walker:
    def __init__(self, init_pos : ArrayLike, gradient_field : ArrayLike) -> None:
        self._init_pos = jnp.array(init_pos)
        self._gradient_field = jnp.array(gradient_field)

        self.time = 0
        self.position = jnp.copy(self._init_pos)

    def step(self):
        pass
