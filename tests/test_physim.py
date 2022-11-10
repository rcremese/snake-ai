##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-11-10 2:28:05 pm
 # @copyright https://mit-license.org/
 #
import pytest
import pygame
from snake_ai.physim.particle import Particle

class TestParticle:
    initial_pos = (0, 0)
    diffusion_coef = 1
    radius = 1
    particle = Particle(*initial_pos, radius, diffusion_coef)

    def test_step(self):
        x, y = self.particle.position
        self.particle.step()
        x_new, y_new = self.particle.position
        assert x != x_new or y != y_new

    def test_reset(self):
        self.particle.reset(5, 5)
        x, y = self.particle.position
        assert x == 5 and y == 5

    def test_grid_conversion(self):
        self.particle.reset(0.2, 0.6)
        x, y = self.particle.get_grid_position()
        assert x == 0 and y == 1

    def test_collision(self):
        obstacle = pygame.Rect(1, 1, 10, 10)
        assert not self.particle.collide(obstacle)
        obstacle = pygame.Rect(0, 0, 10, 10)
        assert self.particle.collide(obstacle)
        # Edge case where the rectangle touch the particle
        obstacle = pygame.Rect(0.5, 0.5, 10, 10)
        assert self.particle.collide(obstacle)

    def test_collision_list(self):
        pixel = [10, 10]
        rect_1 = pygame.Rect(0, 0, *pixel)
        rect_2 = pygame.Rect(1, 1, *pixel)
        assert self.particle.collideall([rect_1, rect_2])