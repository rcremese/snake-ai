import numpy as np
import pygame
from typing import List

class Particle():
    def __init__(self, x_init : float, y_init : float, radius : int = 1, diff_coef : float = 1) -> None:
        self.reset(x_init, y_init)
        self.radius = radius
        self.diff_coef = diff_coef

    def step(self):
        self.position += self.diff_coef * np.random.randn(2)

    def reset(self, x_init : float, y_init : float):
        self.position = np.array([x_init, y_init], dtype=float)

    def collide(self, obstacle : pygame.Rect) -> bool:
        assert isinstance(obstacle, pygame.Rect), 'Use colideall if you want to test collision against a list'
        ## obvious case where the particle center is inside the obstacle
        if obstacle.collidepoint(*self.position):
            return True
        ## case where the particle might be outside of the obstacle but collide with it
        projection = np.copy(self.position)
        # projection along the x-axis
        if self.position[0] < obstacle.left:
            projection[0] = obstacle.left
        elif self.position[0] > obstacle.right:
            projection[0] = obstacle.right
        # projection along the y-axis
        if self.position[1] < obstacle.top:
            projection[1] = obstacle.top
        elif self.position[1] > obstacle.bottom:
            projection[1] = obstacle.bottom
        # distance check between the projection and the center of the particle
        return (np.linalg.norm(self.position - projection) <= self.radius)

    def collideall(self, obstacles : List[pygame.Rect]) -> bool:
        assert isinstance(obstacles, list), 'Use collide if you want to test collision against one obstacle'
        bounding_box = pygame.Rect(*(self.position - self.radius), self.radius, self.radius)
        # Define a bounding box to check for potential collisions
        for idx in bounding_box.collidelistall(obstacles):
            if self.collide(obstacles[idx]):
                return True
        return False

    def draw(self, canvas):
        pass

    def __repr__(self) -> str:
        return f"{__class__.__name__}({self.position[0]!r},{self.position[1]!r}, {self.radius!r},  {self.diff_coef!r})"

    def get_grid_position(self) -> List:
        return [round(coord) for coord in self.position]