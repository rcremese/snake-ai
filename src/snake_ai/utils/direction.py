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
    NORTH = (0, -1)
    NORTH_EAST = (np.sqrt(2) / 2, -np.sqrt(2) / 2)
    EAST = (1, 0)
    SOUTH_EAST = (np.sqrt(2) / 2, np.sqrt(2) / 2)
    SOUTH = (0, 1)
    SOUTH_WEST = (-np.sqrt(2) / 2, np.sqrt(2) / 2)
    WEST = (-1, 0)
    NORTH_WEST = (-np.sqrt(2) / 2, -np.sqrt(2) / 2)


def get_direction_from_vector(vector: Tuple[int]) -> Direction:
    assert len(vector) == 2
    for direction in Direction:
        if vector == direction.value:
            return direction
    raise ValueError(f"Unknown displacement {vector}")


def get_opposite_direction(direction: Direction) -> Direction:
    if direction == Direction.EAST:
        return Direction.WEST

    if direction == Direction.WEST:
        return Direction.EAST

    if direction == Direction.SOUTH:
        return Direction.NORTH

    if direction == Direction.NORTH:
        return Direction.SOUTH

    if direction == Direction.NORTH_EAST:
        return Direction.SOUTH_WEST

    if direction == Direction.SOUTH_WEST:
        return Direction.NORTH_EAST

    if direction == Direction.NORTH_WEST:
        return Direction.SOUTH_EAST

    if direction == Direction.SOUTH_EAST:
        return Direction.NORTH_WEST

    raise ValueError(f"Unknown direction {direction}")
