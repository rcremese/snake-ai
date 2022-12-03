##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-05-13 6:21:04 pm
 # @copyright https://mit-license.org/
 #
import pygame
import logging
from typing import List
from snake_ai.utils.direction import Direction, get_opposite_direction, get_direction_from_vector
from snake_ai.utils import Colors

BODY_PIXEL_SIZE = 12

class Snake():
    def __init__(self, x : float, y : float, pixel_size : int = 20) -> None:
        self._pixel_size = pixel_size
        self.head = pygame.Rect(x, y, self._pixel_size, self._pixel_size)
        self.body = [self.head.move(-self._pixel_size, 0), self.head.move(-2*self._pixel_size, 0)]
        self._size = len(self.body) + 1
        self.direction = Direction.RIGHT

    def draw(self, display, head_color : Colors = Colors.BLUE2, body_color : Colors = Colors.WHITE):
        # draw the head and eye
        pygame.draw.rect(display, head_color.value, self.head)
        # draw the body
        for pt in self.body:
            pygame.draw.rect(display, head_color.value, pt)
            # draw the body of the snake, in order to count the parts
            pygame.draw.rect(display, body_color.value, pt.inflate(-0.25 * self._pixel_size, -0.25 * self._pixel_size))

    def move(self, direction : Direction):
        # Change direction when going in the opposite
        if direction == get_opposite_direction(self.direction):
            logging.debug(f'Chosen direction {direction} is the opposite of the current direction {self.direction}. The snake turns back !')
            self.go_back()
        else:
            self.direction = direction
        # move all parts of the snake
        if self.direction == Direction.RIGHT:
            new_head = self.head.move(self._pixel_size, 0)
        elif self.direction == Direction.LEFT:
            new_head = self.head.move(-self._pixel_size, 0)
        elif self.direction == Direction.DOWN:
            new_head = self.head.move(0, self._pixel_size)
        elif self.direction == Direction.UP:
            new_head = self.head.move(0, -self._pixel_size)
        else:
            raise ValueError(f'Unknown direction {direction}')
        # Check the intersetion of the new head with the rest of the body
        self.body.insert(0, self.head)
        self.head = new_head
        self.body.pop()

    def go_back(self):
        self.direction = self.get_direction_from_head_position(invert=True)
        self.body.insert(0, self.head) # insert the head in front of the list
        self.head = self.body.pop()
        self.body.reverse()

    def grow(self):
        pre_tail = self.body[-2]
        tail = self.body[-1]
        new_tail = tail.move(pre_tail.x - tail.x, pre_tail.y - tail.y)
        self.body.append(new_tail)
        self._size += 1

    def get_direction(self):
        return self.direction

    def get_direction_from_head_position(self, invert : bool = False) -> Direction:
        if invert:
            disp_vect = ((self.body[-1].x - self.body[-2].x) // self._pixel_size, (self.body[-1].y - self.body[-2].y) // self._pixel_size)
        else:
            disp_vect = ((self.head.x - self.body[0].x) // self._pixel_size, (self.head.y - self.body[0].y) // self._pixel_size)
        return get_direction_from_vector(disp_vect)

    def collide_with_itself(self):
        return self.head.collidelist(self.body) != -1

    def collide_with_obstacle(self, obstacle : pygame.Rect):
        return self.head.colliderect(obstacle)

    def collide_with_obstacles(self, obstacle_list : List[pygame.Rect]):
        return self.head.collidelist(obstacle_list) != -1

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter([self.head , *self.body])

    def __next__(self):
        return next([self.head , *self.body])

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Snake):
            raise TypeError(f"Can only make comparision with Snake instances, not {type(__o)} ")
        if len(__o) != self._size:
            return False
        return __o.head == self.head and __o.body == self.body

class SnakeAI(Snake):
    def move_from_action(self, action : int):
        """move the snake given the action

        Args:
            action (int): int that reprensent move possibilities in snake frame
                0 -> turn left
                1 -> continue in the same direction
                2 -> turn right
        """
        clock_wise = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        idx = clock_wise.index(self.direction)
        if action == 0:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
        elif action == 1:
            new_dir = clock_wise[idx] # no change
        elif action == 2:
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else:
            raise ValueError(f'Unknown action {action}')
        # finaly move
        self.move(new_dir)

class SnakeHuman(Snake):
    def move_from_action(self, action : int):
        if action == 0:
            new_dir = Direction.LEFT
        elif action == 1:
            new_dir = Direction.UP
        elif action == 2:
            new_dir = Direction.RIGHT
        elif action == 3:
            new_dir = Direction.DOWN
        else:
            raise ValueError(f'Action need to be in range [0, 4]. Get {action}')
        self.move(new_dir)