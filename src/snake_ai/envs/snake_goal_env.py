##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-10-28 3:35:34 pm
 # @copyright https://mit-license.org/
 #
from snake_ai.envs import Snake2dEnv
import gym
import gym.spaces

from snake_ai.envs.utils import Reward
import pygame
from collections import OrderedDict
import numpy as np
from typing import Dict, Optional, Tuple

class SnakeGoalEnv(Snake2dEnv):
    def __init__(self, render_mode=None, width: int = 20, height: int = 20, nb_obstacles: int = 0, pixel: int = 20):
        super().__init__(render_mode, width, height, nb_obstacles, pixel)
        self.observation_space = gym.spaces.Dict({
            "observation" : gym.spaces.Box(low=np.zeros((self._nb_obs, 2)), high=np.repeat([self.window_size], self._nb_obs, axis=0),
             shape=(self._nb_obs, 2)), # point cloud in game coordinates
            "achieved_goal" : gym.spaces.Box(low=np.zeros(2), high=np.array(self.window_size), shape=(2,), dtype=int), # snake head position
            "desired_goal" : gym.spaces.Box(low=np.zeros(2), high=np.array(self.window_size), shape=(2,), dtype=int), # food position
        })

    def step(self, action : int) -> Tuple[np.ndarray, float, bool, Dict]:
        # Map the action (element of {0,1,2}) to the direction we walk in
        self.snake.move_from_action(action)
        # Do not compute collision lines when the snake is out of bound
        if not self._is_outside():
            self._compute_collision_lines()
        # compute next state + associated reward
        obs = self._get_obs()
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])

        # Continue episode if the snake eat the food
        truncated = self.snake.head.colliderect(self._food)
        if truncated:
            self.snake.grow()
            self.score += 1
            self._place_food()
        # game ends if the snake is outside
        terminated = self._is_outside()
        return obs, reward, terminated, self.info

    def compute_reward(self, achieved_goal : np.ndarray, desired_goal : np.ndarray, info : Optional[Dict] = None) -> int:
        # head center collide with food
        if np.array_equal(achieved_goal, desired_goal):
            return Reward.FOOD.value
        # head center is outside or collide with an obstacle
        pixel = pygame.Rect(*achieved_goal, self._pixel_size, self._pixel_size)
        if (pixel.collidelist(self.obstacles) != -1) or self._is_outside(pixel):
            return Reward.COLLISION.value
        # else return the collision free reward
        return Reward.COLLISION_FREE.value

    def _get_obs(self) -> OrderedDict:
        return OrderedDict({
            'observation' : super()._get_obs(),
            'achieved_goal' : np.array(self.snake.head.center),
            'desired_goal' : np.array(self._food.center),
        })