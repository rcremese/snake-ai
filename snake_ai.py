##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-06-17 4:32:46 pm
 # @copyright https://mit-license.org/
 #

import pygame
from typing import List, Tuple
from snake import Snake
from utils import Direction, get_opposite_direction

class SnakeAI(Snake):
    def __init__(self, x: float, y: float, pixel_size: int = 20, body_pixel: int = 12) -> None:
        super().__init__(x, y, pixel_size, body_pixel)
        self._collision_lines = self.get_collision_lines()

    def get_collision_lines(self, ):
        opposite_direction = get_opposite_direction(self._direction)
        
        up = ((self.head.centerx, self.head.centery), (self.head.centerx, 0))
         
        pass

    def line_collision(self, obstacles : List[pygame.Rect]):
        pass 

    def possible_obstacles_collision(self, obstacles : List[pygame.Rect]) -> List[bool]:
        left, front, right = self.__get_bounding_boxes()
        possible_collision_left = left.collidelist(obstacles) != -1
        possible_collision_front = front.collidelist(obstacles) != 1
        possible_collision_right = right.collidelist(obstacles) != 1
        return [possible_collision_left, possible_collision_front, possible_collision_right]

    def possible_self_collision(self) -> List[bool]:
        if self._size <= 3:
            return [False, False, False]
        left, front, right = self.__get_bounding_boxes()
        possible_collision_left = left.collidelist(self.body[3:]) != -1
        possible_collision_front = front.collidelist(self.body[3:]) != 1
        possible_collision_right = right.collidelist(self.body[3:]) != 1
        return [possible_collision_left, possible_collision_front, possible_collision_right]

    def __get_bounding_boxes(self, size: int = 1) -> Tuple[pygame.Rect]:
        if self._direction == Direction.UP:
            left = self.head.move(-self._pixel_size, 0).inflate(1, 2 * self._pixel_size)
            front = self.head.move(0, -self._pixel_size).inflate(2 * self._pixel_size, 1)
            right = self.head.move(self._pixel_size, 0).inflate(1, 2 * self._pixel_size)
        elif self._direction == Direction.DOWN:
            left = self.head.move(self._pixel_size, 0).inflate(1, 2 * self._pixel_size)
            front = self.head.move(0, self._pixel_size).inflate(2 * self._pixel_size, 1)
            right = self.head.move(-self._pixel_size, 0).inflate(1, 2 * self._pixel_size)
        elif self._direction == Direction.RIGHT:
            left = self.head.move(0, -self._pixel_size).inflate(2 * self._pixel_size, 1)
            front = self.head.move(self._pixel_size, 0).inflate(1, 2 * self._pixel_size)
            right = self.head.move(0, self._pixel_size).inflate(2 * self._pixel_size, 1)
        elif self._direction == Direction.LEFT:
            left = self.head.move(0, self._pixel_size).inflate(2 * self._pixel_size, 1)
            front = self.head.move(-self._pixel_size, 0).inflate(1, 2 * self._pixel_size)
            right = self.head.move(0, -self._pixel_size).inflate(2 * self._pixel_size, 1)
        else:
            raise ValueError(f'Unknown direction {self._direction}')
        return left, front, right

    # def _move(action):
    #     pass