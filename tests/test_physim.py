##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-11-10 2:28:05 pm
 # @copyright https://mit-license.org/
 #
import pygame
from snake_ai.physim.particle import Particle

class TestParticle:
    initial_pos = (0, 0)
    radius = 1
    particle = Particle(*initial_pos, radius)

    def test_move(self):
        self.particle.move(1, 1)
        x_new, y_new = self.particle._position
        assert x_new == 1 or y_new == 1

    def test_reset(self):
        self.particle.reset(5, 5)
        x, y = self.particle._position
        assert x == 5 and y == 5

    def test_grid_conversion(self):
        self.particle.reset(0.2, 0.6)
        x, y = self.particle.get_grid_position()
        assert x == 0 and y == 1

    def test_collision(self):
        pixel = [10, 10]
        obstacle_1 = pygame.Rect(2, 2, *pixel)
        assert not self.particle.collide(obstacle_1)
        obstacle_2 = pygame.Rect(-1, -1, *pixel)
        assert self.particle.collide(obstacle_2)
        # Edge case where the rectangle touch the particle but the
        obstacle_3 = pygame.Rect(0.5, 0.5, *pixel)
        assert self.particle.collide(obstacle_3)

    def test_collision_list(self):
        pixel = [10, 10]
        rect_1 = pygame.Rect(0, 0, *pixel)
        rect_2 = pygame.Rect(1, 1, *pixel)
        assert self.particle.collide_any([rect_1, rect_2])
        rect_3 = pygame.Rect(-2, -2, 1, 1)
        rect_4 = pygame.Rect(2, 2, 1, 1)
        assert not self.particle.collide_any([rect_3, rect_4])

    def test_update(self):
        new_position = (5, 5)
        x, y = self.particle.update(*new_position)
        assert x == 5 and y==5