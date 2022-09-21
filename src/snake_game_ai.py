##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-05-16 11:46:54 am
 # @copyright https://mit-license.org/
 #
import pygame
import random
from typing import Dict, List

import torch
import logging
from snake_game import SnakeGame
from utils import Direction, Line, get_opposite_direction
from snake_ai import SnakeAI

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN = (0,255,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
GREY = (150, 150, 150)

PIXEL_SIZE = 20
OBSTACLE_SIZE_RANGE = (2, 4)
SPEED = 100
PERCENTAGE = 0
TIME_LIMIT = 10

class SnakeGameAI(SnakeGame): 
    def play_step(self, action : torch.Tensor):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self.snake.move_from_action(action) # update the head
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > TIME_LIMIT*len(self.snake):
            game_over = True
            reward = -10
            if self.frame_iteration > TIME_LIMIT*len(self.snake):
                reward *=-2
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.snake.head.colliderect(self.food):
            self.score += 1
            reward = 10
            self.snake.grow()
            self._place_food()

        # 5. update ui and clock
        self._update_ui()
        pygame.display.flip()

        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
