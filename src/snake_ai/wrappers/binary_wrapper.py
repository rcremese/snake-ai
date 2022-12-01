##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-10-28 2:47:09 pm
 # @copyright https://mit-license.org/
 #
# TODO : Remove the file (replaced by SnakeClassicEnv)
import logging
import pygame
from snake_ai.envs import Snake2dEnv
import numpy as np

import gym
import gym.spaces

from snake_ai.utils import Direction

class BinaryWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        assert isinstance(env, Snake2dEnv)
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.MultiBinary(12)

    def observation(self, observation):
        snake_head = self.env.snake.head
        snake_direction = self.env.snake.direction
        food_position = self.env._food

        up = snake_head.move(0, -self.env.pixel_size)
        down = snake_head.move(0, self.env.pixel_size)
        left = snake_head.move(-self.env.pixel_size, 0)
        right = snake_head.move(self.env.pixel_size, 0)

        observations = np.array([
            ## Neighbors collision
            # UP
            (up.collidelist(self.env.obstacles) != -1) or self.env._is_outside(up),
            # RIGHT
            (right.collidelist(self.env.obstacles) != -1) or self.env._is_outside(right),
            # DOWN
            (down.collidelist(self.env.obstacles) != -1) or self.env._is_outside(down),
            # LEFT
            (left.collidelist(self.env.obstacles) != -1) or self.env._is_outside(left),

            ## Snake direction
            # UP
            snake_direction == Direction.UP,
            # RIGHT
            snake_direction == Direction.RIGHT,
            # DOWN
            snake_direction == Direction.DOWN,
            # LEFT
            snake_direction == Direction.LEFT,

            ## Food position
            # UP
            food_position.y < snake_head.y,
            # RIGHT
            food_position.x > snake_head.x,
            # DOWN
            food_position.y > snake_head.y,
            # LEFT
            food_position.x < snake_head.x,
        ], dtype=int)

        return observations

        # # neighbors collision
        # if ((up.collidelist(self.env.obstacles) != -1) or self.env._is_outside(up)):
        #     observations[0] = 1
        # if (right.collidelist(self.env.obstacles) != -1) or self.env._is_outside(right):
        #     observations[1] = 1
        # if (down.collidelist(self.env.obstacles) != -1) or self.env._is_outside(down):
        #     observations[2] = 1
        # if (left.collidelist(self.env.obstacles) != -1) or self.env._is_outside(left):
        #     observations[3] = 1

        # # snake direction
        # if self.env.snake.direction == Direction.UP:
        #     observations[4] = 1
        # if self.env.snake.direction == Direction.RIGHT:
        #     observations[5] = 1
        # if self.env.snake.direction == Direction.DOWN:
        #     observations[6] = 1
        # if self.env.snake.direction == Direction.LEFT:
        #     observations[7] = 1

        # ## food position
        # # UP
        # if self.env.food.y < snake_head.y:
        #     observations[8] = 1
        # # RIGHT
        # if self.env.food.x > snake_head.x:
        #     observations[9] = 1
        # # DOWN
        # if self.env.food.y > snake_head.y:
        #     observations[10] = 1
        # # LEFT
        # if self.env.food.x < snake_head.x:
        #     observations[11] = 1
        # logging.debug(f"Observations : {observations}")

    @property
    def info(self):
        return self.env.info