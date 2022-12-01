from snake_ai.envs.snake_base_env import SnakeBaseEnv
from snake_ai.utils import Reward

import gym
import gym.spaces
import numpy as np
import pygame

from collections import OrderedDict
from typing import Optional, Dict, Tuple

class HerGoalWrapper(gym.Wrapper):
    def __init__(self, env : SnakeBaseEnv):
        super().__init__(env)
        self.env : SnakeBaseEnv
        self.observation_space = gym.spaces.Dict({
            "observation" : self.env.observation_space,
            "achieved_goal" : gym.spaces.Box(low=np.zeros(2), high=np.array(self.window_size), shape=(2,), dtype=int), # snake head position
            "desired_goal" : gym.spaces.Box(low=np.zeros(2), high=np.array(self.window_size), shape=(2,), dtype=int), # food position
        })

    def step(self, action):
        obs, _, terminated, info = super().step(action)
        observation = self.observation(obs)
        reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'], info)
        return observation, reward, terminated, info

    def compute_reward(self, achieved_goal : np.ndarray, desired_goal : np.ndarray, info : Optional[Dict] = None) -> int:
        assert achieved_goal.shape == desired_goal.shape
        # head center collide with food
        if np.array_equal(achieved_goal, desired_goal):
            return Reward.FOOD.value
        # check if the head center is outside or collide with an obstacle
        if any(achieved_goal < [0, 0]) or any(achieved_goal > self.env.window_size):
            return Reward.COLLISION.value
        for obstacle in self.env.obstacles:
            if obstacle.collidepoint(*achieved_goal):
                return Reward.COLLISION.value
        # else return the collision free reward
        return Reward.COLLISION_FREE.value

    def step(self, action : int) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, terminated, info = super().step(action)
        return obs, reward, terminated, self.info

    def observation(self, observation):
        obs = {
            "observation" : observation,
            "achieved_goal" : self.env.snake.head.center,
            "desired_goal" : self.env.food.center,
        }
        return obs
