##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-11-10 2:28:07 pm
 # @copyright https://mit-license.org/
 #
from typing import Dict, List, Tuple
import pygame
from snake_ai.envs import SnakeClassicEnv, SnakeAI, GridWorld
from snake_ai.utils.line import Line
from snake_ai.utils import Direction
from snake_ai.wrappers.relative_position_wrapper import RelativePositionWrapper
import numpy as np
from pathlib import Path
from phi import flow
import json

class TestGridWorld():
    w, h, pix = 5, 5, 10
    nb_obs, max_obs_size = 5, 2

    grid_world = GridWorld(w, h, pix, nb_obs, max_obs_size)

    def test_init(self):
        grid_world = GridWorld(self.w, self.h, self.pix, self.nb_obs, self.max_obs_size)
        assert grid_world.width == self.w
        assert grid_world.height == self.h
        assert grid_world.nb_obs == self.nb_obs
        assert grid_world._max_obs_size == self.max_obs_size

    def test_reset(self):
        obs, info = self.grid_world.reset()

    def test_step(self):
        pass
    
    def test_render(self):
        pass
    
    def test_close(self):
        pass

    def test_observation(self):
        pass

    def test_free_position(self):
        pass
class TestSnakeClassicalEnv():
    w, h, pix = 5, 5, 10
    nb_obs = 2

    def test_reset(self):
        env = SnakeClassicEnv(render_mode=None, width=self.w, height=self.h, nb_obstacles=self.nb_obs, pixel=self.pix)

        env.reset()
        assert env.snake == SnakeAI(10, 10, pixel=self.pix)

    def test_write(self, tmp_path):
        env = SnakeClassicEnv(width=self.w, height=self.h, nb_obstacles=self.nb_obs, pixel=self.pix)
        # Test simple environment write
        output_path : Path = tmp_path.joinpath('env_write.json')
        env.write(output_path)
        with open(output_path, 'r') as file:
            env_dict = json.load(file)
        assert env_dict == {"width": self.w, "height": self.h, "pixel": self.pix, "seed": 0, "nb_obstacles": self.nb_obs,"max_obs_size": 3, "render_mode": "None"}
        env.write(output_path, detailed=True)
        with open(output_path, 'r') as file:
            env_dict = json.load(file)
        assert env_dict == {
            "width": self.w,
            "height": self.h,
            "pixel": self.pix,
            "seed": 0,
            "nb_obstacles": self.nb_obs,
            "max_obs_size": 3,
            "render_mode": "None",

            }

    def test_load(self):
        outputpath = Path(__file__).parent.joinpath('data', 'environment.json').resolve(strict=True)
        env = SnakeClassicEnv.load(outputpath)
        assert env.height == 10
        assert env.width == 10
        assert env._pixel_size == 10
        assert env._max_obs_size == 2
        assert env.nb_obstacles == 2
        assert env.render_mode is None
        assert env.food == Circle(5, 5, 5)
        assert env.snake == SnakeAI(50, 50, 10)
        assert env.obstacles == [Rectangle(10,10,10,10), Rectangle(40, 40, 10, 10)]

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

from snake_ai.envs.geometry import Rectangle, Circle
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
        assert phiflow_rect == flow.Box(x=(self.x_0, self.x_0 + self.w_0), y=(self.y_0, self.y_0 + self.h_0))

    def test_dict_conversion(self):
        rect = Rectangle(self.x_0, self.y_0, self.w_0, self.h_0)
        dictionary = {'left' : self.x_0, 'right' : self.x_0 + self.w_0, 'top' : self.y_0, 'bottom' : self.y_0 + self.h_0}
        rect_dict = rect.to_dict()
        assert rect_dict == dictionary

        new_rect = Rectangle.from_dict(dictionary)
        assert new_rect == rect

class TestSphere:
    x, y = 0, 1
    radius = 1
    sphere = Circle(x, y, radius)

    def test_dict_conversion(self):
        dictionary = self.sphere.to_dict()
        assert dictionary == {'center' : [self.x, self.y], 'radius': self.radius}
        new_sphere = Circle.from_dict(dictionary)
        assert new_sphere == self.sphere