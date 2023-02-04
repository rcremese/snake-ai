##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-11-10 2:28:07 pm
 # @copyright https://mit-license.org/
 #
from typing import Dict, List, Tuple
import pygame
from snake_ai.envs import SnakeClassicEnv, SnakeAI
from snake_ai.utils.line import Line
from snake_ai.utils import Direction
from snake_ai.wrappers.relative_position_wrapper import RelativePositionWrapper
import numpy as np
import phi
class TestSnakeClassicalEnv():
    w, h, pix = 5, 5, 10
    nb_obs = 10

    def test_reset(self):
        env = SnakeClassicEnv(render_mode=None, width=self.w, height=self.h, nb_obstacles=self.nb_obs, pixel=self.pix)
        env.seed()

        env.reset()
        assert env.snake == SnakeAI(10, 10, pixel_size=self.pix)

    def test_step(self):
        pass

    def test_properties(self):
        pass

    def test_place_food(self):
        pass

    def test_place_obstacel(self):
        pass

    def test_is_outside(self):
        pass

    def test_collide_with_obstacles(slef):
        pass

    def test_collide_with_snake_body(self):
        pass

    def test_collisions(self):
        pass

from snake_ai.envs.geometry import Rectangle
class TestRectangle:
    x_0, y_0, w_0, h_0 = 0, 0, 1, 1
    x_1, y_1, w_1, h_1 = -5, -4, -1, 2

    def test_init(self):
        rect_1 = Rectangle(self.x_0, self.y_0, self.w_0, self.h_0)
        rect_2 = Rectangle(pygame.Rect(self.x_0, self.y_0, self.w_0, self.h_0))
        assert rect_1 == rect_2

    def test_phiflow_conversion(self):
        rect = Rectangle(self.x_0, self.y_0, self.w_0, self.h_0)
        phiflow_rect = rect.to_phiflow()
        assert phiflow_rect == phi.geom.Box(x=(self.x_0, self.x_0 + self.w_0), y=(self.y_0, self.y_0 + self.h_0))

    def test_dict_conversion(self):
        rect = Rectangle(self.x_0, self.y_0, self.w_0, self.h_0)
        rect_dict = rect.to_dict()
        assert rect_dict ==  {'left' : self.x_0, 'right' : self.x_0 + self.w_0, 'top' : self.y_0, 'bottom' : self.y_0 + self.h_0}
