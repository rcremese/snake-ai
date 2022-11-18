##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Particle that interacts with the environment
 # @desc Created on 2022-11-10 10:17:42 pm
 # @copyright https://mit-license.org/
 #
import numpy as np
import pygame
from snake_ai.utils import Colors
from typing import List, Union

class Particle():
    def __init__(self, x_init : float, y_init : float, radius : int = 1) -> None:
        self.reset(x_init, y_init)
        self.radius = radius

    def reset(self, x_init : Union[float, int], y_init : Union[float, int]):
        assert isinstance(x_init, (float, int)) and isinstance(y_init, (float, int))
        self._position = np.array([x_init, y_init], dtype=float)

    def collide(self, obstacle : pygame.Rect) -> bool:
        assert isinstance(obstacle, pygame.Rect), 'Use colideall if you want to test collision against a list'
        ## obvious case where the particle center is inside the obstacle
        if obstacle.collidepoint(*self._position):
            return True
        ## case where the particle might be outside of the obstacle but collide with it
        projection = np.copy(self._position)
        # projection along the x-axis
        if self._position[0] < obstacle.left:
            projection[0] = obstacle.left
        elif self._position[0] > obstacle.right:
            projection[0] = obstacle.right
        # projection along the y-axis
        if self._position[1] < obstacle.top:
            projection[1] = obstacle.top
        elif self._position[1] > obstacle.bottom:
            projection[1] = obstacle.bottom
        # distance check between the projection and the center of the particle
        return (np.linalg.norm(self._position - projection) <= self.radius)

    def collide_any(self, obstacles : List[pygame.Rect]) -> bool:
        assert isinstance(obstacles, list), 'Use collide if you want to test collision against one obstacle'
        bounding_box = pygame.Rect(*(self._position - self.radius), 2 * self.radius, 2 * self.radius)
        # Define a bounding box to check for potential collisions
        for idx in bounding_box.collidelistall(obstacles):
            if self.collide(obstacles[idx]):
                return True
        return False

    def is_inside(self, rect : pygame.Rect, center : bool = False) -> bool:
        if center:
            return rect.collidepoint(*self._position)
        lower_pos = self._position - self.radius
        upper_pos = self._position + self.radius
        return lower_pos[0] >= rect.left and lower_pos[1] >= rect.top and upper_pos[0] <= rect.right and upper_pos[1] <= rect.bottom

    def draw(self, canvas : pygame.Surface):
        pygame.draw.circle(canvas, Colors.MIDDLE_GREEN.value, self._position, self.radius)

    def move(self, dx : Union[float, int], dy : Union[float, int]):
        assert isinstance(dx, (float, int)) and isinstance(dy, (float, int))
        self._position += np.array([dx, dy])

    def update_position(self, x : Union[float, int], y : Union[float, int]):
        self.reset(x, y)

    def get_grid_position(self) -> List:
        return [round(coord) for coord in self._position]

    def __eq__(self, other: __build_class__) -> bool:
        return np.array_equal(self._position, other._position) and self.radius == other.radius

    def __repr__(self) -> str:
        return f"{__class__.__name__}({self._position[0]!r},{self._position[1]!r}, {self.radius!r})"