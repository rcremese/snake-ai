##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-11-10 2:28:07 pm
 # @copyright https://mit-license.org/
 #
from typing import Dict, List, Tuple
import pygame
from snake_ai.envs.snake_2d_env import Snake2dEnv
from snake_ai.utils.line import Line
from snake_ai.utils import Direction
from snake_ai.wrappers.binary_wrapper import BinaryWrapper
from snake_ai.wrappers.relative_position_wrapper import RelativePositionWrapper
import numpy as np

class TestSnakeEnv():
    w, h, pix= 5, 5, 10

    def test_env(self):
        env = Snake2dEnv(width=self.w, height=self.h, nb_obstacles=0, pixel=self.pix)
        obs = env.reset()
        snake_head_pos = env.snake.head.center
        assert env.window_size == (self.w * self.pix, self.h * self.pix)
        assert snake_head_pos == ((self.w * self.pix) // 2, (self.h * self.pix) // 2),  f"Snake head position {snake_head_pos}"

        playground = pygame.Rect((0,0), env.window_size)
        assert playground.center == snake_head_pos
        assert playground.right == self.w * self.pix


        col_line : Line = env.collision_lines[Direction.RIGHT]
        assert  col_line.start == snake_head_pos, f"Collision line start at {col_line.start}"
        assert col_line.end == (self.w * self.pix, (self.h * self.pix) // 2),  f"Collision line ends at {col_line.end}"

    def test_observations(self):
        env = Snake2dEnv(width=self.w, height=self.h, pixel=self.pix)
        wrapped_env = RelativePositionWrapper(env)
        observations = wrapped_env.reset()
        info = wrapped_env.info
        directions = wrapped_env._get_list_direction(info['snake_direction'])
        passing_matrix = wrapped_env._get_passing_matrix(info['snake_direction'])
        assert info['snake_direction'] == Direction.RIGHT
        assert info['collision_lines'][Direction.LEFT] is None
        assert len(observations) == 8

        dist_center_border = 2.5 * env.pixel_size
        assert np.array_equal(observations[0], np.array([0, dist_center_border])), f"The observable for direction {directions[0]} is {observations[0]}" # RIGHT
        assert np.array_equal(observations[1], np.array([dist_center_border, dist_center_border])), f"The observable for direction {directions[1]} is {observations[1]}" # DOWN_RIGHT
        assert np.array_equal(observations[2], np.array([dist_center_border, 0])), f"The observable for direction {directions[2]} is {observations[2]}" # DOWN
        assert np.array_equal(observations[3], np.array([dist_center_border, -dist_center_border])), f"The observable for direction {directions[3]} is {observations[3]}" # DOWN_LEFT
        assert np.array_equal(observations[4], np.array([-dist_center_border, -dist_center_border])), f"The observable for direction {directions[4]} is {observations[4]}" # UP_LEFT
        assert np.array_equal(observations[5], np.array([-dist_center_border, 0])), f"The observable for direction {directions[4]} is {observations[5]}" # UP
        assert np.array_equal(observations[6], np.array([-dist_center_border, dist_center_border])), f"The observable for direction {directions[6]} is {observations[6]}" # UP_RIGHT
        food = Line(info['snake_head'], info['food'])
        local_food = np.matmul(passing_matrix, food.to_vector())
        assert np.array_equal(observations[7], local_food), f"The food vector for food located at {info['food']} is {observations[7]}" # FOOD

    def test_passing_matrix(self):
        env = Snake2dEnv(width=5, height=5)
        wrapped_env = RelativePositionWrapper(env)
        for direction in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
            passing_matrix = wrapped_env._get_passing_matrix(direction)
            if direction == Direction.UP:
                assert np.array_equal(passing_matrix, np.array([[1, 0], [0, -1]]))
            if direction == Direction.RIGHT:
                assert np.array_equal(passing_matrix, np.array([[0, 1], [1, 0]]))
            if direction == Direction.DOWN:
                assert np.array_equal(passing_matrix, np.array([[-1, 0], [0, 1]]))
            if direction == Direction.LEFT:
                assert np.array_equal(passing_matrix, np.array([[0, -1], [-1, 0]]))

    def test_binary_obs(self):
        env = Snake2dEnv(width=self.w, height=self.h, pixel=self.pix)
        wrapped_env = BinaryWrapper(env)
        wrapped_env.seed(0)
        ## obs = [direction (Up-Right-Down-Left), danger (URDL), food (URDL)]
        obs = wrapped_env.reset()
        # food center is at position (45,5)
        # snake head center is at position (25, 25)
        assert np.array_equal(obs, np.array([0, 0, 0 , 0, 0, 1, 0, 0, 1, 1, 0, 0])), f"Binary observation {obs}"
        # continue in the current direction (right)
        obs, _, _, _ = wrapped_env.step(1)
        # snake head center is at position (35, 25)
        assert np.array_equal(obs, np.array([0, 0, 0 , 0, 0, 1, 0, 0, 1, 1, 0, 0]))
        # continue in the current direction (right)
        obs, _, _, _ = wrapped_env.step(1)
        # snake head center is at position (45, 25)
        # danger at right !
        assert np.array_equal(obs, np.array([0, 1, 0 , 0, 0, 1, 0, 0, 1, 0, 0, 0]))
        # go down
        obs, _, _, _ = wrapped_env.step(2)
        # snake head center is at position (45, 35)
        # danger at right !
        assert np.array_equal(obs, np.array([0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]))
        # continue in the current direction (down)
        obs, _, _, _ = wrapped_env.step(1)
        # snake head center is at position (45, 45)
        # danger at right and bottom !
        assert np.array_equal(obs, np.array([0, 1, 1 , 0, 0, 0, 1, 0, 1, 0, 0, 0]))
        # turn left
        obs, _, _, _ = wrapped_env.step(2)
        # snake head center is at position (35, 45)
        # danger at the bottom !
        assert np.array_equal(obs, np.array([0, 0, 1 , 0, 0, 0, 0, 1, 1, 1, 0, 0]))
        # go outside !
        obs, _, terminated, _ = wrapped_env.step(0)
        assert terminated
        # snake head center is at position (35, 55)
        # danger everywhere except up !
        assert np.array_equal(obs, np.array([0, 1, 1 , 1, 0, 0, 1, 0, 1, 1, 0, 0]))

    def _get_food_position(self, info : Dict[str, Tuple]) -> List[bool]:
        food_position = [
            info['food'][1] < info['snake_head'][1], #UP
            info['food'][0] > info['snake_head'][0], #RIGHT
            info['food'][1] >= info['snake_head'][1], #DOWN
            info['food'][0] <= info['snake_head'][0], #LEFT
        ]
        return food_position


