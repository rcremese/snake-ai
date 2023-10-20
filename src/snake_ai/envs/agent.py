from abc import ABCMeta, abstractmethod
from snake_ai.envs.geometry import Rectangle
from snake_ai.utils import Direction, errors
from typing import List, Tuple
import pygame


class Agent(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None:
        self._direction: Direction = None
        self._position: Rectangle = None
        self.pixel: int = None

    ## Public methods
    @abstractmethod
    def move(self, direction: Direction):
        raise NotImplementedError()

    @abstractmethod
    def move_from_action(self, action: int):
        raise NotImplementedError()

    @abstractmethod
    def draw(self, canvas: pygame.Surface):
        raise NotImplementedError()

    ## Properties
    @property
    def position(self) -> Rectangle:
        "Agent position in the environment, represented by a rectangle"
        return self._position

    @position.setter
    def position(self, rect: Rectangle):
        if rect.width != self.pixel or rect.height != self.pixel:
            raise ValueError(
                "Can not assign a new position with different pixel size."
                + f"Current pixel size  : {self.pixel}. Rectangle pixel size : {rect.width, rect.height}"
            )
        self._position = rect

    @property
    def direction(self) -> Direction:
        "Direction of the agent in the environment"
        return self._direction

    @direction.setter
    def direction(self, direction: Direction):
        if not isinstance(direction, Direction):
            raise TypeError(f"{direction} is not an instance of Direction.")
        if direction.name not in ["NORTH", "EAST", "SOUTH", "WEST"]:
            raise errors.UnknownDirection(
                f"The direction {direction} is unknown. Available directions are : 'NORTH', 'EAST', 'SOUTH', 'WEST'."
            )
        self._direction = direction

    @property
    @abstractmethod
    def neighbours(self) -> List[Rectangle]:
        "Neighbours of the current position of the agent"
        raise NotImplementedError()

    ## Dunder methods
    @abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError()
