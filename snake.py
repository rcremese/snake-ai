##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-05-13 6:21:04 pm
 # @copyright https://mit-license.org/
 #
import pygame
from pyparsing import opAssoc
from traitlets import directional_link
from utils import Direction, get_opposite_direction

class Snake:
    def __init__(self, x : float, y : float, pixel_size : int = 20) -> None:
        self.pixel_size = pixel_size
        self.head = pygame.Rect(x, y, self.pixel_size, self.pixel_size)
        self.body = [self.head, self.head.move(-self.pixel_size, 0), self.head.move(-2*self.pixel_size, 0)]
        self.size = len(self.body)
        self.direction = Direction.RIGHT

    def go_back(self):
        self.body.reverse()
        self.head = self.body[0]

    def move(self, direction : Direction):
        # Change direction when going in the opposite 
        if direction == get_opposite_direction(self.direction):
            self.go_back()
              
        self.direction = direction
        # move all parts of the snake
        if direction == Direction.RIGHT:
            new_head = self.head.move(self.pixel_size, 0)
        elif direction == Direction.LEFT:
            new_head = self.head.move(-self.pixel_size, 0)
        elif direction == Direction.DOWN:
            new_head = self.head.move(0, self.pixel_size)
        elif direction == Direction.UP:
            new_head = self.head.move(0, -self.pixel_size)
        else:
            raise ValueError(f'Unknown direction {direction}')
        # Check the intersetion of the new head with the rest of the body
        if new_head.colliderect(self.body[1]):
            self.move(get_opposite_direction(direction))
        else:
            # Move foreward
            self.body.insert(0, new_head)
            self.head = new_head
            self.body.pop()

    def grow(self):
        pre_tail = self.body[-2]
        tail = self.body[-1]
        new_tail = tail.move(pre_tail.x - tail.x, pre_tail.y - tail.y)
        self.body.append(new_tail)
        self.size += 1
        
    def collide_with_itself(self):
        return self.head.collidelist(self.body[1:]) != -1
        
    def collide_with_obstacle(self, obstacle : pygame.Rect):
        return self.head.colliderect(obstacle)

    def collide_with_obstacles(self, obstacle_list : list[pygame.Rect]):
        return self.head.collidelist(obstacle_list) != -1

    def __len__(self):
        return len(self.body)
        
    def __iter__(self):
        return iter(self.body)
        
    def __next__(self):
        return next(self.body)

class Snake(Snake):
    def possible_obstacles_collision(self, obstacles : list[pygame.Rect]) -> list[bool]:
        left, front, right = self.__get_bounding_boxes()
        possible_collision_left = left.collidelist(obstacles) != -1
        possible_collision_front = front.collidelist(obstacles) != 1
        possible_collision_right = right.collidelist(obstacles) != 1
        return [possible_collision_left, possible_collision_front, possible_collision_right]

    def possible_self_collision(self) -> list[bool]:
        if self.size <= 3:
            return [False, False, False]
        left, front, right = self.__get_bounding_boxes()
        possible_collision_left = left.collidelist(self.body[3:]) != -1
        possible_collision_front = front.collidelist(self.body[3:]) != 1
        possible_collision_right = right.collidelist(self.body[3:]) != 1
        return [possible_collision_left, possible_collision_front, possible_collision_right]

    def __get_bounding_boxes(self, size: int = 1) -> tuple[pygame.Rect]:
        if self.direction == Direction.UP:
            left = self.head.move(-self.pixel_size, 0).inflate(1, 2 * self.pixel_size)
            front = self.head.move(0, -self.pixel_size).inflate(2 * self.pixel_size, 1)
            right = self.head.move(self.pixel_size, 0).inflate(1, 2 * self.pixel_size)
        elif self.direction == Direction.DOWN:
            left = self.head.move(self.pixel_size, 0).inflate(1, 2 * self.pixel_size)
            front = self.head.move(0, self.pixel_size).inflate(2 * self.pixel_size, 1)
            right = self.head.move(-self.pixel_size, 0).inflate(1, 2 * self.pixel_size)
        elif self.direction == Direction.RIGHT:
            left = self.head.move(0, -self.pixel_size).inflate(2 * self.pixel_size, 1)
            front = self.head.move(self.pixel_size, 0).inflate(1, 2 * self.pixel_size)
            right = self.head.move(0, self.pixel_size).inflate(2 * self.pixel_size, 1)
        elif self.direction == Direction.LEFT:
            left = self.head.move(0, self.pixel_size).inflate(2 * self.pixel_size, 1)
            front = self.head.move(-self.pixel_size, 0).inflate(1, 2 * self.pixel_size)
            right = self.head.move(0, -self.pixel_size).inflate(2 * self.pixel_size, 1)
        else:
            raise ValueError(f'Unknown direction {self.direction}')
        return left, front, right

    # def _move(action):
    #     pass