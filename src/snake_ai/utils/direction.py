##
# @author Robin CREMESE <robin.cremese@gmail.com>
# @file Directions used by the snake in pygame
# @desc Created on 2022-11-10 10:52:33 pm
# @copyright https://mit-license.org/
#
import enum
import numpy as np
from typing import Tuple

class Direction(enum.Enum):
    UP = (0,-1)
    UP_RIGHT = (np.sqrt(2) / 2, - np.sqrt(2) / 2)
    RIGHT = (1,0)
    DOWN_RIGHT = (np.sqrt(2) / 2, np.sqrt(2) / 2)
    DOWN = (0,1)
    DOWN_LEFT = (-np.sqrt(2) / 2, np.sqrt(2) / 2)
    LEFT = (-1, 0)
    UP_LEFT = (-np.sqrt(2) / 2, -np.sqrt(2) / 2)

def get_direction_from_vector(vector : Tuple[int]) -> Direction:
    assert len(vector) == 2
    for direction in Direction:
        if vector == direction.value:
            return direction
    raise ValueError(f'Unknown displacement {vector}')

def get_opposite_direction(direction : Direction) -> Direction:
    if direction == Direction.RIGHT:
        return Direction.LEFT

    if direction == Direction.LEFT:
        return Direction.RIGHT

    if direction == Direction.DOWN:
        return Direction.UP

    if direction == Direction.UP:
        return Direction.DOWN

    if direction == Direction.UP_RIGHT:
        return Direction.DOWN_LEFT

    if direction == Direction.DOWN_LEFT:
        return Direction.UP_RIGHT

    if direction == Direction.UP_LEFT:
        return Direction.DOWN_RIGHT

    if direction == Direction.DOWN_RIGHT:
        return Direction.UP_LEFT

    raise ValueError(f'Unknown direction {direction}')
