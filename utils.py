##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-05-14 12:22:25 am
 # @copyright https://mit-license.org/
 #
from enum import Enum
import pygame
import time

from snake_game_V1 import BLUE1

class Direction(Enum):
    RIGHT = (-1,0)
    LEFT = (1,0)
    UP = (0,-1)
    DOWN = (0,1)

def get_opposite_direction(direction):
    if direction == Direction.RIGHT:
        return Direction.LEFT
    elif direction == Direction.LEFT:
        return Direction.RIGHT
    elif direction == Direction.DOWN:
        return Direction.UP
    elif direction == Direction.UP:
        return Direction.DOWN
    else:
        raise ValueError(f'Unknown direction {direction}')

# RED = (255,0, 0)
# WHITE = (255,255,255)
# BLUE1 = (0,0,255)
# display = pygame.display.set_mode((100, 100))
# pygame.display.set_caption('Snake')
# pix_size = 20
# point = pygame.Rect(0,50, pix_size, pix_size)
# pygame.draw.rect(display, RED, point)
# point.move_ip(50,0)
# pygame.draw.rect(display, RED, point.inflate(1,2*pix_size))
# pygame.draw.rect(display, WHITE, point)
# pygame.draw.rect(display, BLUE1, point.move(-20,0))
# pygame.display.flip()
# time.sleep(5)
            