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
