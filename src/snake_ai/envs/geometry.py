from abc import ABCMeta, abstractmethod
from typing import Dict, Any, List, Optional
import pygame
from snake_ai.utils.types import Numerical
from phi.jax import flow
from snake_ai.utils import Colors
import numpy as np

class Geometry(metaclass=ABCMeta):
    def __repr__(self) -> str:
        return "Abstract class defining an obstacle"

    @abstractmethod
    def to_phiflow(self):
        raise NotImplementedError()

    @abstractmethod
    def to_dict(self) -> Dict:
        raise NotImplementedError()

    def _check_is_numerical(self, value : Any, name : str):
        if not isinstance(value, (float, int)):
            raise TypeError(f"Expected float or int value for {name}, get {value}")

class Rectangle(pygame.Rect, Geometry):
    def to_phiflow(self) -> flow.Box:
        return flow.Box(x=(self.left, self.right), y=(self.top, self.bottom))

    def to_dict(self) -> Dict:
        return {'left' : self.left, 'right' : self.right, 'top' : self.top, 'bottom' : self.bottom}

    def draw(self, canvas : pygame.Surface, color : Colors = Colors.RED):
        pygame.draw.rect(canvas, color.value, self)

class Sphere(Geometry):
    def __init__(self, x_init : Numerical, y_init : Numerical, radius : Numerical) -> None:
        self._center = np.array([x_init, y_init], dtype=float)
        if radius <= 0:
            raise ValueError(f"Radius should be > 0, get {radius}")
        self.radius = radius

    def _collide_rect(self, rect : Rectangle) -> bool:
        assert isinstance(rect, Rectangle), 'Use colideall if you want to test collision against a list'
        ## obvious case where the particle center is inside the obstacle
        if rect.collidepoint(*self._center):
            return True
        ## case where the sphere might be outside of the obstacle but collide with it
        projection = np.copy(self._center)
        # projection along the x-axis
        if self._center[0] < rect.left:
            projection[0] = rect.left
        elif self._center[0] > rect.right:
            projection[0] = rect.right
        # projection along the y-axis
        if self._center[1] < rect.top:
            projection[1] = rect.top
        elif self._center[1] > rect.bottom:
            projection[1] = rect.bottom
        # distance check between the projection and the center of the particle
        return (np.linalg.norm(self._center - projection) <= self.radius)

    def _collide_sphere(self, sphere) -> bool:
        assert isinstance(sphere, Sphere), f"Expected to check collision with a sphere, not {type(sphere)}"
        return np.linalg.norm(self._center - sphere._center) < self.radius + sphere.radius

    def collide(self, obstacle : Geometry):
        if isinstance(obstacle, Rectangle):
            return self._collide_rect(obstacle)
        elif isinstance(obstacle, Sphere):
            return self._collide_sphere(obstacle)
        else:
            raise TypeError(f"Can not check collisions for instance of {type(obstacle)}")

    def collide_any(self, obstacles : List[Geometry], return_idx : bool) -> bool and Optional[int]:
        assert isinstance(obstacles, list), 'Use collide if you want to test collision against one obstacle'
        for i, obstacle in enumerate(obstacles):
            if self.collide(obstacle):
                if return_idx:
                    return True, i
                return True
        return False

    def draw(self, canvas : pygame.Surface, color : Colors = Colors.MIDDLE_GREEN):
        pygame.draw.circle(canvas, color.value, self._center, self.radius)

    def to_phiflow(self) -> flow.Box:
        return flow.Sphere(x=self._center[0], y=self._center[1], radius=self.radius)

    def to_dict(self) -> Dict:
        return {'center' : self._center.tolist(), 'radius' : self.radius}

    def __repr__(self) -> str:
        return f"{__class__.__name__}({self._center[0]!r},{self._center[1]!r}, {self.radius!r})"
