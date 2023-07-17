##
# @author Robin CREMESE <robin.cremese@gmail.com>
# @file Description
# @desc Created on 2022-05-13 6:21:04 pm
# @copyright https://mit-license.org/
#
import logging
import pygame
from typing import List, Dict, Tuple
from snake_ai.envs.geometry import Rectangle
from snake_ai.utils.direction import (
    Direction,
    get_opposite_direction,
    get_direction_from_vector,
)
from snake_ai.envs.agent import Agent
from snake_ai.utils import Colors, errors
import numpy as np


def get_direction_from_positions(head: Rectangle, neck: Rectangle) -> Direction:
    """Get the direction from the neck to the head using the vector difference (head - neck).

    Args:
        head (Rectangle): Rectangle that represent the head
        neck (Rectangle): Rectangle that represent the neck, i.e : the first body part after the head.

    Returns:
        Direction: Direction of the normal vector pointing from the neck to the head.
    """
    assert (head.width == neck.width) and (
        head.height == neck.height
    ), f"The 2 arguments head {head} and neck {neck} need to have the same width and height."
    disp_vect = ((head.x - neck.x) // head.width, (head.y - neck.y) // head.height)
    return get_direction_from_vector(disp_vect)


def check_snake_positions(positions: List[Tuple[int]]):
    # type check
    if not isinstance(positions, list):
        raise TypeError(
            f"Positions of the snake parts need to be a list of tuples containing x and y positions. Get {type(positions)}"
        )
    if len(positions) < 2:
        raise errors.ShapeError("Can not initialize a snake with less than 2.")
    if not all(isinstance(pos, tuple) and len(pos) == 2 for pos in positions):
        raise TypeError(
            f"Positions should be tuple of 2 integers representing x and y positions. Get {positions}"
        )
    # value check
    vector_positions = np.array(
        positions
    )  # Make sure the displacement vector between 2 positions has L1 norm equal to 1
    diff = np.linalg.norm(vector_positions[:-1] - vector_positions[1:], ord=1, axis=1)
    if not np.array_equal(diff, np.ones(len(positions) - 1)):
        raise errors.ConfigurationError(
            f"The given positions does not form a continuous path."
        )


class Snake(Agent):
    def __init__(self, positions: List[Tuple[int]], pixel: int) -> None:
        assert pixel > 0, f"Pixel need to be a positive integer. Get {pixel}"
        self.pixel = int(pixel)
        check_snake_positions(positions)
        # Set head and body positons
        self.head = Rectangle(
            positions[0][0] * pixel, positions[0][1] * pixel, pixel, pixel
        )
        self.body: List[Rectangle] = []
        for x, y in positions[1:]:
            self.body.append(
                Rectangle(x * self.pixel, y * self.pixel, self.pixel, self.pixel)
            )
        # Make one last check on position configuration.
        if self.collide_with_itself():
            raise errors.ConfigurationError(
                f"The given list of positions collide with itself. "
                + "Please provide non overlapping positions."
            )
        # Set direction from body position
        self.direction = get_direction_from_positions(self.head, self.body[0])

    # Public methods
    def draw(self, canvas: pygame.Surface):
        # draw the head
        pygame.draw.rect(canvas, Colors.BLUE2.value, self.head)
        # draw the body
        for pt in self.body:
            pygame.draw.rect(canvas, Colors.BLUE2.value, pt)
            # draw the body of the snake, in order to count the parts
            pygame.draw.rect(
                canvas,
                Colors.WHITE.value,
                pt.inflate(-0.25 * self.pixel, -0.25 * self.pixel),
            )

    def move(self, direction: Direction):
        if direction == get_opposite_direction(self.direction):
            raise errors.CollisionError(
                f"The current direction {direction} make the snake to collide with itself."
            )
        self.direction = direction
        # move the new head with respect to the direction
        if self.direction == Direction.EAST:
            new_head = self.head.move(self.pixel, 0)
        elif self.direction == Direction.WEST:
            new_head = self.head.move(-self.pixel, 0)
        elif self.direction == Direction.SOUTH:
            new_head = self.head.move(0, self.pixel)
        elif self.direction == Direction.NORTH:
            new_head = self.head.move(0, -self.pixel)
        # update all properties of the snake given the new position
        self.body.insert(0, self.head)
        self.head = new_head
        self.body.pop()

    def move_from_action(self, action: int):
        """move the snake given the action

        Args:
            action (int): int that reprensent move possibilities in snake frame
                0 -> turn left
                1 -> continue in the same direction
                2 -> turn right
        """
        assert isinstance(
            action, int
        ), f"Action need to be an integer in the range [0,2]. Get {action}."
        if action not in range(3):
            raise errors.InvalidAction(
                f"Action need to be an integer in the range [0,2]. Get {action}."
            )
        clock_wise = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        idx = clock_wise.index(self.direction)
        if action == 0:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d
        elif action == 1:
            new_dir = clock_wise[idx]  # no change
        elif action == 2:
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:
            raise ValueError(f"Unknown action {action}")
        # finaly move
        self.move(new_dir)

    def grow(self):
        if len(self.body) == 1:
            pre_tail = self.head
        else:
            pre_tail = self.body[-2]
        tail = self.body[-1]
        # new_tail = tail.move(pre_tail.x - tail.x, pre_tail.y - tail.y)
        new_tail = tail.move(tail.x - pre_tail.x, tail.y - pre_tail.y)
        self.body.append(new_tail)

    def collide_with_itself(self):
        return self.head.collidelist(self.body) != -1

    ## Properties
    @property
    def neighbours(self) -> List[Rectangle]:
        north = self.head.move(0, -self.pixel)
        east = self.head.move(self.pixel, 0)
        south = self.head.move(0, self.pixel)
        west = self.head.move(-self.pixel, 0)
        # Return the 3 neighbours in clockwise order, from left to right
        if self.direction == Direction.NORTH:
            return west, north, east
        if self.direction == Direction.EAST:
            return north, east, south
        if self.direction == Direction.SOUTH:
            return east, south, west
        if self.direction == Direction.WEST:
            return south, west, north
        raise errors.UnknownDirection(
            f"The current direction {self.direction} is unknown for finding neighbours."
        )

    @Agent.position.getter
    def position(self) -> Rectangle:
        return self.head

    @position.setter
    def position(self, rect: Rectangle):
        if rect.width != self.pixel or rect.height != self.pixel:
            raise ValueError(
                "Can not assign a new position with different pixel size."
                + f"Current pixel size  : {self.pixel}. Rectangle pixel size : {rect.width, rect.height}"
            )
        disp = (rect.x - self.head.x, rect.y - self.head.y)
        self.head = rect
        for body_part in self.body:
            body_part.move_ip(disp[0], disp[1])

    ## Private methods
    # def _get_direction_from_position(self) -> Direction:
    #     disp_vect = ((self.head.x - self.body[0].x) // self.pixel, (self.head.y - self.body[0].y) // self.pixel)
    #     return get_direction_from_vector(disp_vect)

    def to_dict(self) -> Dict[str, int]:
        return {"x": self.head.x, "y": self.head.y, "pixel": self.pixel}

    @classmethod
    def from_dict(cls, dictionary: Dict[str, int]):
        keys = ["x", "y", "pixel"]
        if any([key not in dictionary.keys() for key in keys]):
            raise ValueError(
                f"Input dictonary need to contain the following keys : 'x', 'y', 'pixel'. Get {dictionary.keys()}"
            )
        return cls(dictionary["x"], dictionary["y"], dictionary["pixel"])

    # def collide_with_obstacle(self, obstacle : pygame.Rect):
    #     return self.head.colliderect(obstacle)

    # def collide_with_obstacles(self, obstacle_list : List[pygame.Rect]):
    #     return self.head.collidelist(obstacle_list) != -1

    ## Dunder methods
    def __len__(self):
        return len(self.body) + 1

    def __iter__(self):
        return iter([self.head, *self.body])

    def __next__(self):
        return next([self.head, *self.body])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Snake):
            raise TypeError(
                f"Can only make comparision with Snake instances, not {type(other)} "
            )
        if len(other) != self.__len__():
            return False
        return (
            other.head == self.head
            and other.body == self.body
            and other.direction == self.direction
        )

    def __repr__(self) -> str:
        return f"{__class__.__name__}(head={self.head!r}, body={self.body!r}, pixel={self.pixel})"


class BidirectionalSnake(Snake):
    def go_back(self):
        self.body.insert(0, self.head)  # insert the head in front of the list
        self.head = self.body.pop()
        self.body.reverse()
        self.direction = get_direction_from_positions(self.head, self.body[0])

    def move_from_action(self, action: int):
        assert action in range(
            4
        ), f"Action need to be an integer in the range [0,3]. Get {action}."
        if action in range(3):
            super().move_from_action(action)
        else:
            self.go_back()

    @Snake.neighbours.getter
    def neighbours(self) -> List[Rectangle]:
        left, front, right = super().neighbours
        if len(self.body) == 1:
            tail_disp = (self.body[0].x - self.head.x) // self.pixel, (
                self.body[0].y - self.head.y
            ) // self.pixel
        else:
            tail_disp = (self.body[-1].x - self.body[-2].x), (
                self.body[-1].y - self.body[-2].y
            )
        back = self.body[-1].move(*tail_disp)
        return left, front, right, back


class SnakeHuman(BidirectionalSnake):
    def move_from_action(self, action: int):
        if action not in range(4):
            raise ValueError(
                f"Unable to map direction with the input value {action}. Action need to be an integer in range [0,3]"
            )
        if action == 0:
            self.move(Direction.NORTH)
        elif action == 1:
            self.move(Direction.EAST)
        elif action == 2:
            self.move(Direction.SOUTH)
        elif action == 3:
            self.move(Direction.WEST)

    def move(self, direction: Direction):
        if direction == get_opposite_direction(self.direction):
            logging.debug(
                f"Chosen direction {direction} is the opposite of the current direction {self.direction}. The snake turns back !"
            )
            self.go_back()
        super().move(direction)
