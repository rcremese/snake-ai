##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-05-13 6:21:04 pm
 # @copyright https://mit-license.org/
 #
import pygame
from utils import Direction

PIXEL_SIZE = 20
class Snake:
    def __init__(self, x : float, y : float, pixel_size : int = 20) -> None:
        # top = y - PIXEL_SIZE / 2
        # left = x - PIXEL_SIZE / 2
        self.head = pygame.Rect(x - PIXEL_SIZE / 2, y - PIXEL_SIZE / 2, PIXEL_SIZE, PIXEL_SIZE)
        self.body = [self.head, self.head.move(-PIXEL_SIZE, 0), self.head.move(-2*PIXEL_SIZE, 0)]
        self.direction = Direction.RIGHT

    def go_back(self):
        self.body.reverse()
        self.head = self.body[0]

    def move(self, direction : Direction):
        # TODO : Change direction when colliding
        self.direction = direction
        # move all parts of the snake
        if direction == Direction.RIGHT:
            new_head = self.head.move(PIXEL_SIZE, 0)
        elif direction == Direction.LEFT:
            new_head = self.head.move(-PIXEL_SIZE, 0)
        elif direction == Direction.DOWN:
            new_head = self.head.move(0, PIXEL_SIZE)
        elif direction == Direction.UP:
            new_head = self.head.move(0, -PIXEL_SIZE)
        # Move foreward
        self.body.insert(0, new_head)
        self.head = new_head
        self.body.pop()

    def grow(self):
        pre_tail = self.body[-2]
        tail = self.body[-1]
        new_tail = tail.move(pre_tail.x - tail.x, pre_tail.y - tail.y)
        self.body.append(new_tail)

    def collide_with_borders(self, width : float, hight : float) -> bool:
        return (self.head.x + PIXEL_SIZE > width) or (self.head.x < 0) or (self.head.y + PIXEL_SIZE > hight) or (self.head.y < 0)
        
    def collide_with_itself(self):
        return self.head.collidelist(self.body[1:]) != -1
        
    def collide_with_obstacle(self, obstacle_list : list):
        return self.head.collidelist(obstacle_list) != -1

    def __iter__(self):
        return iter(self.body)
        
    def __next__(self):
        return next(self.body)
