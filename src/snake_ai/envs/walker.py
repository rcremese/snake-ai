from snake_ai.envs.agent import Agent
from snake_ai.envs.geometry import Rectangle
from snake_ai.utils import Direction, Colors, errors
from typing import List
import pygame

class Walker2D(Agent):
    def __init__(self, x: int, y : int, pixel : int) -> None:
        if pixel < 1:
            raise ValueError(f"Pixel argument can not be lower that 1. Get {pixel}")
        self.pixel = int(pixel)
        self._position = Rectangle(x * self.pixel, y * self.pixel, self.pixel, self.pixel)
        self._direction = Direction.NORTH

    ## Public methods
    def move(self, direction : Direction):
        self.direction = direction
        if direction == Direction.NORTH:
            self._position.move_ip(0, -self.pixel)
        elif direction == Direction.EAST:
            self._position.move_ip(self.pixel, 0)
        elif direction == Direction.SOUTH:
            self._position.move_ip(0, self.pixel)
        elif direction == Direction.WEST:
            self._position.move_ip(-self.pixel, 0)
        else:
            raise errors.UnknownDirection(f"Unknown direction {direction}. Expected directions : 'NORTH', 'EAST', 'SOUTH', 'WEST'.")

    def move_from_action(self, action : int):
        if action not in range(4):
            raise errors.InvalidAction(f"Unable to map direction with the input value {action}. Action need to be an integer in range [0,3]")
        if action == 0:
            self.move(Direction.NORTH)
        elif action == 1:
            self.move(Direction.EAST)
        elif action == 2:
            self.move(Direction.SOUTH)
        elif action == 3:
            self.move(Direction.WEST)

    def draw(self, canvas: pygame.Surface):
        pygame.draw.rect(canvas, Colors.BLUE2.value, self.position)

    # Properties
    @property
    def neighbours(self) -> List[Rectangle]:
        "Neighbours of the current position of the agent"
        top = self.position.move(0, -self.pixel)
        right = self.position.move(self.pixel, 0)
        bottom = self.position.move(0, self.pixel)
        left = self.position.move(-self.pixel, 0)
        return top, right, bottom, left

    ## Dunder methods
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Walker2D):
            raise TypeError(f"Can not compare instance of Walker2D with {type(other)}")
        return self.direction == other.direction and self.position == other.position

    def __repr__(self) -> str:
        return f"{__class__.__name__}(position={self.position!r}, direction={self.direction}, pixel={self.pixel})"