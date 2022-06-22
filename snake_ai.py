##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-06-17 4:32:46 pm
 # @copyright https://mit-license.org/
 #

import torch
from typing import List, Tuple
from snake import Snake
from utils import Direction, get_opposite_direction

class SnakeAI(Snake):
    def move_from_action(self, action : torch.Tensor) -> None:
        """move the snake given the action

        Args:
            action (torch.Tensor): tensor that reprensent move possibilities 
                Tensor([1,0,0]) -> turn left
                Tensor([0,1,0]) -> continue in the same direction
                Tensor([0,0,1]) -> turn rigth
        """
        # TODO : include possibility to do back turn 
        clock_wise = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        idx = clock_wise.index(self.snake._direction)

        if action.equal(torch.tensor([1, 0, 0])):
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
        elif action.equal(torch.tensor([0, 1, 0])):
            new_dir = clock_wise[idx] # no change
        elif action.equal(torch.tensor([0, 0, 1])):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else:
            raise ValueError(f'Unknown action {action}')
        # finaly move
        self.move(new_dir)

    # def possible_obstacles_collision(self, obstacles : List[pygame.Rect]) -> List[bool]:
    #     left, front, right = self.__get_bounding_boxes()
    #     possible_collision_left = left.collidelist(obstacles) != -1
    #     possible_collision_front = front.collidelist(obstacles) != 1
    #     possible_collision_right = right.collidelist(obstacles) != 1
    #     return [possible_collision_left, possible_collision_front, possible_collision_right]

    # def possible_self_collision(self) -> List[bool]:
    #     if self._size <= 3:
    #         return [False, False, False]
    #     left, front, right = self.__get_bounding_boxes()
    #     possible_collision_left = left.collidelist(self.body[3:]) != -1
    #     possible_collision_front = front.collidelist(self.body[3:]) != 1
    #     possible_collision_right = right.collidelist(self.body[3:]) != 1
    #     return [possible_collision_left, possible_collision_front, possible_collision_right]

    # def __get_bounding_boxes(self, size: int = 1) -> Tuple[pygame.Rect]:
    #     if self._direction == Direction.UP:
    #         left = self.head.move(-self._pixel_size, 0).inflate(1, 2 * self._pixel_size)
    #         front = self.head.move(0, -self._pixel_size).inflate(2 * self._pixel_size, 1)
    #         right = self.head.move(self._pixel_size, 0).inflate(1, 2 * self._pixel_size)
    #     elif self._direction == Direction.DOWN:
    #         left = self.head.move(self._pixel_size, 0).inflate(1, 2 * self._pixel_size)
    #         front = self.head.move(0, self._pixel_size).inflate(2 * self._pixel_size, 1)
    #         right = self.head.move(-self._pixel_size, 0).inflate(1, 2 * self._pixel_size)
    #     elif self._direction == Direction.RIGHT:
    #         left = self.head.move(0, -self._pixel_size).inflate(2 * self._pixel_size, 1)
    #         front = self.head.move(self._pixel_size, 0).inflate(1, 2 * self._pixel_size)
    #         right = self.head.move(0, self._pixel_size).inflate(2 * self._pixel_size, 1)
    #     elif self._direction == Direction.LEFT:
    #         left = self.head.move(0, self._pixel_size).inflate(2 * self._pixel_size, 1)
    #         front = self.head.move(-self._pixel_size, 0).inflate(1, 2 * self._pixel_size)
    #         right = self.head.move(0, -self._pixel_size).inflate(2 * self._pixel_size, 1)
    #     else:
    #         raise ValueError(f'Unknown direction {self._direction}')
    #     return left, front, right

    # def _move(action):
    #     pass