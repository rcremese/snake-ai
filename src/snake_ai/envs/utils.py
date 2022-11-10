##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-05-14 12:22:25 am
 # @copyright https://mit-license.org/
 #
import enum
from typing import Optional, Union, Tuple, List
import logging
from pathlib import Path
import numpy as np

FONT_PATH = Path(__file__).parents[1].joinpath('graphics', 'arial.ttf').resolve(strict=True)

class Colors(enum.Enum):
    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    GREEN = (0, 255, 0)
    BLUE1 = (0, 0, 255)
    BLUE2 = (0, 100, 255)
    BLACK = (0, 0, 0)
    GREY = (150, 150, 150)

class Reward(enum.Enum):
    FOOD = 10
    COLLISION = -10
    COLLISION_FREE = -1

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
    logging.debug(f'input vector : {vector}')
    for direction in Direction:
        if vector == direction.value:
            return direction
    raise ValueError(f'Unknown displacement {vector}')

def get_opposite_direction(direction : Direction) -> Direction:
    logging.debug(f'input direction {direction}')
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
