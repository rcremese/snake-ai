##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-05-14 12:22:25 am
 # @copyright https://mit-license.org/
 #
from enum import Enum
from typing import Union, Tuple, List
import logging
import pygame
import numpy as np

from snake_game_V1 import BLUE1

class Line():
    def __init__(self, start : Tuple[int], end : Tuple[int]) -> None:
        self.start = start
        self.end = end
        self.length = np.linalg.norm(np.array(end, dtype=int) - np.array(start, dtype=int))

    # def __init__(self, x1 : int, y1 : int, x2 : int, y2 : int) -> None:
    #     self.start = (x1, y1)
    #     self.end = (x2, y2)
    #     self.length = np.linalg.norm(np.array(self.end, dtype=int) - np.array(self.start, dtype=int))
    def __repr__(self) -> str:
        return f'Line({self.start}, {self.end})'

    def intersect(self, rect : pygame.Rect) -> Tuple[Tuple[int]]:
        return rect.clipline(self.start, self.end)

    def __len__(self):
        return self.length

    def draw(self, display : pygame.display, line_color : Tuple[int], point_color : Tuple[int]):
        pygame.draw.line(display, line_color, self.start, self.end)
        pygame.draw.circle(display, point_color, self.end, radius=5)

    def intersect_obstacles(self, rect_list : List[pygame.Rect]) -> Union[Tuple[int], None]:
        min_distance = self.length
        starting_point = np.array(self.start, dtype=int)
        point_intersect = None

        for rect in rect_list:
            intersection = self.intersect(rect)
            if intersection:
                # find shortest distance between starting point and possible intersection points
                distances = np.linalg.norm([starting_point - np.array(end_point, dtype=int) for end_point in intersection], axis=1)
                index = np.argmin(distances)
                if distances[index] < min_distance:
                    min_distance = distances[index]
                    point_intersect = intersection[index]
        return point_intersect

class Direction(Enum):
    UP = (0,-1)
    UP_RIGHT = (1, -1)
    RIGHT = (1,0)
    DOWN_RIGHT = (1, 1)
    DOWN = (0,1)
    DOWN_LEFT = (-1, 1)
    LEFT = (-1, 0)
    UP_LEFT = (-1, -1)

def get_direction_from_vector(vector : Tuple[int]):
    logging.debug(f'input vector : {vector}')
    if vector == (1, 0):
        direction = Direction.RIGHT
    elif vector == (-1, 0):
        direction =  Direction.LEFT
    elif vector == (0, -1):
        direction =  Direction.UP
    elif vector == (0, 1):
        direction =  Direction.DOWN
    else:
        raise ValueError(f'Unknown displacement {vector}')    
    logging.debug(f'corresponding direction {direction}')
    return direction

def get_opposite_direction(direction):
    logging.debug(f'input direction {direction}')
    if direction == Direction.RIGHT:
        opposite_dir = Direction.LEFT

    elif direction == Direction.LEFT:
        opposite_dir = Direction.RIGHT

    elif direction == Direction.DOWN:
        opposite_dir = Direction.UP

    elif direction == Direction.UP:
        opposite_dir = Direction.DOWN
    else:
        raise ValueError(f'Unknown direction {direction}')
    logging.debug(f'corresponding opposite direction {opposite_dir}')
    return opposite_dir
