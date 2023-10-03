##
# @author  <robin.cremese@gmail.com>
# @file Implementation of the geometry module containing rectangle and circle classes
# @desc Created on 2023-04-04 10:37:29 am
# @copyright MIT License
#
from abc import ABCMeta, abstractmethod
from typing import Dict, Any, List, Optional
import pygame
from snake_ai.utils.types import Numerical
from snake_ai.utils import Colors
import numpy as np


class Geometry(metaclass=ABCMeta):
    def __repr__(self) -> str:
        return "Abstract class defining an obstacle"

    @abstractmethod
    def to_dict(self) -> Dict[str, int]:
        raise NotImplementedError()

    @abstractmethod
    def from_dict(cls, dictionary: Dict[str, int]):
        raise NotImplementedError()

    def _check_is_numerical(self, value: Any, name: str):
        if not isinstance(value, (float, int)):
            raise TypeError(f"Expected float or int value for {name}, get {value}")


class Rectangle(pygame.Rect, Geometry):
    def to_dict(self) -> Dict:
        return {
            "left": self.left,
            "right": self.right,
            "top": self.top,
            "bottom": self.bottom,
        }

    def to_circle(self):
        return Circle(self.centerx, self.centery, min(self.height, self.width) / 2)

    def draw(self, canvas: pygame.Surface, color: Colors = Colors.RED):
        pygame.draw.rect(canvas, color.value, self)

    @classmethod
    def from_dict(cls, dictionary: Dict[str, int]):
        keys = ["left", "right", "top", "bottom"]
        if any([key not in dictionary.keys() for key in keys]):
            raise ValueError(
                f"Input dictonary need to contain the following keys : 'left', 'right', 'top', 'bottom'. Get {dictionary.keys()}"
            )
        width = dictionary["right"] - dictionary["left"]
        height = dictionary["bottom"] - dictionary["top"]
        return cls(dictionary["left"], dictionary["top"], width, height)


class Circle(Geometry):
    def __init__(self, x_init: Numerical, y_init: Numerical, radius: Numerical) -> None:
        self.center = np.array([x_init, y_init], dtype=float)

        if radius <= 0:
            raise ValueError(f"Radius should be > 0, get {radius}")
        self.radius = radius

    ## Public methods
    def collide(self, obstacle: Geometry):
        if isinstance(obstacle, Geometry):
            return self._collide_rect(obstacle)
        elif isinstance(obstacle, Circle):
            return self._collide_sphere(obstacle)
        else:
            raise TypeError(
                f"Can not check collisions for instance of {type(obstacle)}"
            )

    def collide_any(
        self, obstacles: List[Geometry], return_idx: bool
    ) -> bool and Optional[int]:
        assert isinstance(
            obstacles, list
        ), "Use collide if you want to test collision against one obstacle"
        for i, obstacle in enumerate(obstacles):
            if self.collide(obstacle):
                if return_idx:
                    return True, i
                return True
        return False

    def draw(self, canvas: pygame.Surface, color: Colors = Colors.MIDDLE_GREEN):
        pygame.draw.circle(canvas, color.value, self.center, self.radius)

    def to_dict(self) -> Dict:
        return {
            "x_center": self.center[0],
            "y_center": self.center[1],
            "radius": self.radius,
        }

    def to_rectangle(self) -> Rectangle:
        return Rectangle(*(self.center - self.radius), self.diameter, self.diameter)

    @property
    def diameter(self):
        "Diameter of the circle"
        return 2 * self.radius

    @classmethod
    def from_dict(cls, dictionary: Dict[str, int]) -> object:
        if not {"x_center", "y_center", "radius"}.issubset(dictionary.keys()):
            raise KeyError(
                f"Input dictonary need to contain the following keys : 'x_center', 'y_center','radius'. Get {dictionary.keys()}"
            )
        return cls(dictionary["x_center"], dictionary["y_center"], dictionary["radius"])

    ## Private methods
    def _collide_rect(self, rect: Rectangle) -> bool:
        assert isinstance(
            rect, Rectangle
        ), "Use colideall if you want to test collision against a list"
        ## obvious case where the particle center is inside the obstacle
        if rect.collidepoint(*self.center):
            return True
        ## case where the sphere might be outside of the obstacle but collide with it
        projection = np.copy(self.center)
        # projection along the x-axis
        if self.center[0] < rect.left:
            projection[0] = rect.left
        elif self.center[0] > rect.right:
            projection[0] = rect.right
        # projection along the y-axis
        if self.center[1] < rect.top:
            projection[1] = rect.top
        elif self.center[1] > rect.bottom:
            projection[1] = rect.bottom
        # distance check between the projection and the center of the particle
        return np.linalg.norm(self.center - projection) <= self.radius

    def _collide_sphere(self, sphere) -> bool:
        assert isinstance(
            sphere, Circle
        ), f"Expected to check collision with a sphere, not {type(sphere)}"
        return np.linalg.norm(self.center - sphere.center) < self.radius + sphere.radius

    ## Dunder methods
    def __repr__(self) -> str:
        return f"{__class__.__name__}({self.center[0]!r},{self.center[1]!r}, {self.radius!r})"

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, Circle), f"Can not compare circle with {type(other)}."
        return np.array_equal(self.center, other.center) and self.radius == other.radius


class Cube(Geometry):
    def __init__(
        self, x: int, y: int, z: int, width: int = 1, height: int = 1, depth: int = 1
    ) -> None:
        assert (
            isinstance(x, (int, np.int_))
            and isinstance(y, (int, np.int_))
            and isinstance(z, (int, np.int_))
        ), "x, y and z must be integers"
        self.x = x
        self.y = y
        self.z = z

        assert (
            width > 0 and height > 0 and depth > 0
        ), "Width, height and depth must be positive integers"
        self.width = width
        self.height = height
        self.depth = depth

    def to_dict(self) -> Dict:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
        }

    def from_dict(cls, dictionary: Dict[str, int]):
        return super().from_dict(dictionary)

    def is_inside(self, other: Geometry) -> bool:
        assert isinstance(other, Cube), f"Can not compare cube with {type(other)}."
        return (
            (self.x >= other.x)
            and (self.y >= other.y)
            and (self.z >= other.z)
            and (self.x + self.width <= other.x + other.width)
            and (self.y + self.height <= other.y + other.height)
            and (self.z + self.depth <= other.z + other.depth)
        )

    def contains(self, other: Geometry) -> bool:
        assert isinstance(other, Cube), f"Can not compare cube with {type(other)}."
        return other.is_inside(self)

    @property
    def volume(self) -> int:
        return self.width * self.height * self.depth

    @property
    def center(self) -> np.ndarray:
        return (
            np.array(
                [self.x + self.width, self.y + self.height, self.z + self.depth],
                dtype=float,
            )
            / 2
        )

    @property
    def vertices(self) -> np.ndarray:
        """Get the vertices of the cube.

        The vertices are generated in the following order :\n
        0: (x, y, z)\n
        1: (x + width, y, z)\n
        2: (x, y + height, z)\n
        3: (x, y, z + depth)\n
        4: (x + width, y + height, z)\n
        5: (x + width, y, z + depth)\n
        6: (x, y + height, z + depth)\n
        7: (x + width, y + height, z + depth)\n
        """
        return np.array(
            [
                (self.x, self.y, self.z),
                (self.x + self.width, self.y, self.z),
                (self.x, self.y + self.height, self.z),
                (self.x, self.y, self.z + self.depth),
                (self.x + self.width, self.y + self.height, self.z),
                (self.x + self.width, self.y, self.z + self.depth),
                (self.x, self.y + self.height, self.z + self.depth),
                (self.x + self.width, self.y + self.height, self.z + self.depth),
            ]
        )

    @property
    def edges(self) -> np.ndarray:
        """Returns the edges of the cube.

        The order of the edges is the following : \n
        (0, 1)\n
        (0, 2)\n
        (0, 3)\n
        (1, 4)\n
        (1, 5)\n
        (2, 4)\n
        (2, 6)\n
        (3, 5)\n
        (3, 6)\n
        (4, 7)\n
        (5, 7)\n
        (6, 7)\n
        """
        return np.array(
            [
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 4],
                [1, 5],
                [2, 4],
                [2, 6],
                [3, 5],
                [3, 6],
                [4, 7],
                [5, 7],
                [6, 7],
            ]
        )

    @property
    def min(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def max(self) -> np.ndarray:
        return np.array(
            [self.x + self.width, self.y + self.height, self.z + self.depth]
        )

    ## Dunder methods
    def __repr__(self) -> str:
        return f"{__class__.__name__}({self.x!r},{self.y!r},{self.z!r},{self.width!r},{self.height!r},{self.depth!r})"
