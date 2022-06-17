##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-05-13 6:21:04 pm
 # @copyright https://mit-license.org/
 #
import pygame
from typing import List, Tuple
from utils import Direction, get_opposite_direction

BODY_PIXEL_SIZE = 12

class Snake:
    def __init__(self, x : float, y : float, pixel_size : int = 20, body_pixel : int = 12) -> None:
        self._pixel_size = pixel_size
        self._body_pixel = body_pixel
        self.head = pygame.Rect(x, y, self._pixel_size, self._pixel_size)
        self.body = [self.head.move(-self._pixel_size, 0), self.head.move(-2*self._pixel_size, 0)]
        self._size = len(self.body)
        self._direction = Direction.RIGHT
        
    def draw(self, display, head_color = (255, 255, 255), body_color = (0, 100, 255)):
        move = (self._pixel_size - self._body_pixel) / 2
        # draw the head and eye
        pygame.draw.rect(display, head_color, self.head)
        # TODO : dessiner les yeux
        pygame.draw.rect(display, (0,0,0), self.head)
        # draw the body
        for pt in self.body:
            pygame.draw.rect(display, head_color, pt)
            # draw the body of the snake, in order to count the parts
            pygame.draw.rect(display, body_color, pygame.Rect(pt.x + move, pt.y + move, self._body_pixel, self._body_pixel))
        
    def go_back(self):
        self.body.insert(0, self.head) # insert the head in front of the list
        self.body.reverse()
        self.head = self.body.pop()

    def grow(self):
        pre_tail = self.body[-2]
        tail = self.body[-1]
        new_tail = tail.move(pre_tail.x - tail.x, pre_tail.y - tail.y)
        self.body.append(new_tail)
        self._size += 1
    
    def get_new_direction(self):
        disp_vect = ((self.head.centerx - self.body[0].centerx) // self._pixel_size, (self.head.centery - self.body[0].centery) // self._pixel_size) 
        for direction in Direction:
            if disp_vect == direction.value:
                self._direction = direction
        return self._direction

    def collide_with_itself(self):
        return self.head.collidelist(self.body) != -1
        
    def collide_with_obstacle(self, obstacle : pygame.Rect):
        return self.head.colliderect(obstacle)

    def collide_with_obstacles(self, obstacle_list : List[pygame.Rect]):
        return self.head.collidelist(obstacle_list) != -1

    def __len__(self):
        return len(self.body)
        
    def __iter__(self):
        return iter(self.body)
        
    def __next__(self):
        return next(self.body)

class SnakeHuman(Snake):
    def move(self, direction : Direction):
        # Change direction when going in the opposite 
        if direction == get_opposite_direction(self.direction):
            self.go_back()
                
        self.direction = direction
        # move all parts of the snake
        if direction == Direction.RIGHT:
            new_head = self.head.move(self._pixel_size, 0)
        elif direction == Direction.LEFT:
            new_head = self.head.move(-self._pixel_size, 0)
        elif direction == Direction.DOWN:
            new_head = self.head.move(0, self._pixel_size)
        elif direction == Direction.UP:
            new_head = self.head.move(0, -self._pixel_size)
        else:
            raise ValueError(f'Unknown direction {direction}')
        # Check the intersetion of the new head with the rest of the body
        if new_head.colliderect(self.body[0]):
            self.move(get_opposite_direction(direction))
        else:
            # Move foreward
            self.body.insert(0, self.head)
            self.head = new_head
            self.body.pop()
