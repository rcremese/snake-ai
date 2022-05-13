##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-05-14 12:22:25 am
 # @copyright https://mit-license.org/
 #
from enum import Enum

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

def get_opposite_direction(direction):
    if direction == Direction.RIGHT:
        return Direction.LEFT
    elif direction == Direction.LEFT:
        return Direction.RIGHT
    elif direction == Direction.DOWN:
        return Direction.UP
    elif direction == Direction.UP:
        return direction.DOWN
    else:
        raise ValueError(f'Unknown direction {direction}')