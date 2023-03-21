##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-11-10 2:28:07 pm
 # @copyright https://mit-license.org/
 #
from typing import Dict, List, Tuple
import pygame
from snake_ai.envs import Snake, Walker2D, Rectangle
from snake_ai.utils.line import Line
from snake_ai.utils import Direction, Reward, errors
import numpy as np
from pathlib import Path
from phi import flow
import pytest
import json

from snake_ai.envs.grid_world import GridWorld
class TestGridWorld():
    w, h, pix = 5, 5, 10
    seed = 0
    # Seed 0 configuration
    # ---A-
    # ----G
    # -----
    # -----
    # -----
    agent = Walker2D(3, 0, pixel=pix)
    goal = Rectangle(4 * pix, pix, pix, pix)

    def test_init(self):
        grid_world = GridWorld(self.w, self.h, self.pix, self.seed)
        assert grid_world.width == self.w
        assert grid_world.height == self.h
        assert grid_world.pixel == self.pix
        assert grid_world.window_size == (self.w * self.pix, self.h * self.pix)
        assert grid_world._seed == self.seed
        assert grid_world.render_mode == None
        assert grid_world._screen == None

        with pytest.raises(ValueError):
            grid_world = GridWorld(0, 0, 0)

        with pytest.raises(errors.InitialisationError):
            grid_world.goal


    def test_reset(self):
        grid_world = GridWorld(self.w, self.h, self.pix, self.seed)
        obs, info  = grid_world.reset()
        # Effect of the seed
        assert grid_world.agent == self.agent
        assert grid_world.goal == self.goal
        assert grid_world.obstacles == []
        assert grid_world._truncated == False
        # Check observations
        assert np.array_equal(obs, grid_world.observations)
        assert info == grid_world.info

    def test_observation(self):
        grid_world = GridWorld(self.w, self.h, self.pix, self.seed)
        reset_obs, _ = grid_world.reset()
        assert isinstance(reset_obs, np.ndarray)
        assert reset_obs.shape == (12,)
        assert reset_obs.tolist() == [
            1, 0, 0, 0, # border collisions
            1, 0, 0, 0, # current direction
            0, 1, 1, 0 # goal position
        ]
        # Test configuration
        # G----
        # -----
        # -----
        # -----
        # ----A
        grid_world.goal = Rectangle(0, 0, self.pix, self.pix)
        grid_world.agent = Walker2D(4, 4, pixel=self.pix)

        obs = grid_world.observations
        assert obs.tolist() == [
            0, 1, 1, 0, # border collisions
            1, 0, 0, 0, # current direction
            1, 0, 0, 1 # goal position
        ]

    def test_info(self):
        grid_world = GridWorld(self.w, self.h, self.pix, self.seed)
        _, reset_info = grid_world.reset()
        # Seed 0 configuration
        # ---A-
        # ----G
        # -----
        # -----
        # -----
        assert reset_info == {
            "agent_position": self.agent.position,
            "agent_direction": Direction.NORTH,
            "obstacles": [],
            "goal": self.goal,
            "truncated": False,
        }

    def test_step(self):
        grid_world = GridWorld(self.w, self.h, self.pix, self.seed)
        grid_world.reset()
        # EAST
        # ----A
        # ----G
        # -----
        # -----
        # -----
        obs, reward, terminated, info = grid_world.step(1)
        assert obs.tolist() == [
            1, 1, 0, 0, # border collisions
            0, 1, 0, 0, # current direction
            0, 0, 1, 0 # goal position
        ]
        assert reward == Reward.COLLISION_FREE.value
        assert terminated == False
        assert info["agent_direction"] == Direction.EAST
        assert info["truncated"] == False
        # SOUTH
        # -----
        # ----A
        # --G--
        # -----
        # -----
        obs, reward, terminated, info = grid_world.step(2)
        assert obs.tolist() == [
            0, 1, 0, 0, # border collisions
            0, 0, 1, 0, # current direction
            0, 0, 1, 1 # goal position
        ]
        assert terminated == False
        assert reward == Reward.FOOD.value
        assert info["agent_direction"] == Direction.SOUTH
        assert info["truncated"] == True
        # WEST
        # -----
        # ---A-
        # --G--
        # -----
        # -----
        obs, reward, terminated, info = grid_world.step(3)
        assert obs.tolist() == [
            0, 0, 0, 0, # border collisions
            0, 0, 0, 1, # current direction
            0, 0, 1, 1 # goal position
        ]
        assert terminated == False
        assert reward == Reward.COLLISION_FREE.value
        assert info["agent_direction"] == Direction.WEST
        assert info["truncated"] == False
        # NORTH
        # ---A-
        # -----
        # --G--
        # -----
        # -----
        obs, reward, terminated, info = grid_world.step(0)
        assert obs.tolist() == [
            1, 0, 0, 0, # border collisions
            1, 0, 0, 0, # current direction
            0, 0, 1, 1 # goal position
        ]
        assert terminated == False
        assert reward == Reward.COLLISION_FREE.value
        assert info["agent_direction"] == Direction.NORTH
        assert info["truncated"] == False
        # Collision = NORTH
        # ***A*
        # -----
        # -----
        # --G--
        # -----
        # -----
        obs, reward, terminated, info = grid_world.step(0)
        assert obs.tolist() == [
            1, 1, 0, 1, # border collisions
            1, 0, 0, 0, # current direction
            0, 0, 1, 1 # goal position
        ]
        assert terminated == True
        assert reward == Reward.COLLISION.value

        with pytest.raises(errors.InvalidAction):
            grid_world.step(4)

    def test_free_position(self):
        grid_world = GridWorld(self.w, self.h, self.pix, self.seed)
        grid_world.reset()
        free_pos = np.ones((self.w, self.h), dtype=np.int)
        free_pos[4, 1] = 0
        assert np.array_equal(grid_world._free_position_mask, free_pos)
        assert grid_world.free_positions == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                                             (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
                                             (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
                                             (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
                                             (4, 0), (4, 2), (4, 3), (4, 4)]
        grid_world.goal = Rectangle(0, 0, self.pix, self.pix)
        free_pos[4, 1] = 1
        free_pos[0, 0] = 0
        assert np.array_equal(grid_world._free_position_mask, free_pos)
        assert grid_world.free_positions == [(0, 1), (0, 2), (0, 3), (0, 4),
                                             (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
                                             (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
                                             (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
                                             (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]

from snake_ai.envs.random_obstacles_env import RandomObstaclesEnv
class TestRandomObstaclesEnv:
    w, h, pix = 5, 5, 10
    seed = 0

    def test_obstacles(self):
        grid_world = RandomObstaclesEnv(self.w, self.h, self.pix, seed=self.seed, nb_obs=0)
        # -O---
        # -----
        # --OOO
        # --OOO
        # --OOO
        obstacles = [Rectangle(self.pix, 0, self.pix, self.pix),
                     Rectangle(2 * self.pix, 2 * self.pix, 3 * self.pix, 3 * self.pix)]
        grid_world.obstacles = obstacles
        free_pos = np.ones((self.w, self.h), dtype=int)
        free_pos[1,0] = False
        free_pos[2:,2:] = False
        assert np.array_equal(grid_world._free_position_mask, free_pos)
        assert grid_world.obstacles == obstacles

# class TestSnakeClassicalEnv():
#     w, h, pix = 5, 5, 10
#     nb_obs = 2

#     def test_reset(self):
#         env = SnakeClassicEnv(render_mode=None, width=self.w, height=self.h, nb_obstacles=self.nb_obs, pixel=self.pix)

#         env.reset()
#         assert env.snake == SnakeAI(10, 10, pixel=self.pix)

#     def test_write(self, tmp_path):
#         env = SnakeClassicEnv(width=self.w, height=self.h, nb_obstacles=self.nb_obs, pixel=self.pix)
#         # Test simple environment write
#         output_path : Path = tmp_path.joinpath('env_write.json')
#         env.write(output_path)
#         with open(output_path, 'r') as file:
#             env_dict = json.load(file)
#         assert env_dict == {"width": self.w, "height": self.h, "pixel": self.pix, "seed": 0, "nb_obstacles": self.nb_obs,"max_obs_size": 3, "render_mode": "None"}
#         env.write(output_path, detailed=True)
#         with open(output_path, 'r') as file:
#             env_dict = json.load(file)
#         assert env_dict == {
#             "width": self.w,
#             "height": self.h,
#             "pixel": self.pix,
#             "seed": 0,
#             "nb_obstacles": self.nb_obs,
#             "max_obs_size": 3,
#             "render_mode": "None",

#             }

#     def test_load(self):
#         outputpath = Path(__file__).parent.joinpath('data', 'environment.json').resolve(strict=True)
#         env = SnakeClassicEnv.load(outputpath)
#         assert env.height == 10
#         assert env.width == 10
#         assert env._pixel_size == 10
#         assert env._max_obs_size == 2
#         assert env.nb_obstacles == 2
#         assert env.render_mode is None
#         assert env.food == Circle(5, 5, 5)
#         assert env.snake == SnakeAI(50, 50, 10)
#         assert env.obstacles == [Rectangle(10,10,10,10), Rectangle(40, 40, 10, 10)]

#     def test_step(self):
#         pass

#     def test_properties(self):
#         pass

#     def test_place_food(self):
#         pass

#     def test_place_obstacel(self):
#         pass

#     def test_is_outside(self):
#         pass

#     def test_collide_with_obstacles(slef):
#         pass

#     def test_collide_with_snake_body(self):
#         pass

#     def test_collisions(self):
#         pass