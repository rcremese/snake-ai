##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-10-28 2:47:39 pm
 # @copyright https://mit-license.org/
 #
from collections import OrderedDict
from typing import Optional, Union
import gym
import gym.spaces
from snake_ai.envs.snake_base_env import SnakeBaseEnv
import numpy as np

class DistanceWrapper(gym.ObservationWrapper):
    def __init__(self, env: Union[SnakeBaseEnv, gym.Wrapper]):
        super().__init__(env)

        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_shape = env.observation_space['observation'].shape
            self.observation_space = gym.spaces.Dict({
                "observation" : gym.spaces.Box(low = 0, high = env.max_dist, shape=(obs_shape[0], )), # point cloud in local frame                "achieved_goal" : gym.spaces.Box(low=(0,0), high=self.window_size, shape=(2,0), dtype=int), # snake head position
                "achieved_goal" : gym.spaces.Box(low=np.zeros(2), high=np.array(self.env.window_size), shape=(2,), dtype=int), # snake head position
                "desired_goal" : gym.spaces.Box(low=np.zeros(2), high=np.array(self.env.window_size), shape=(2,), dtype=int), # food position
            })
        elif isinstance(env.observation_space, gym.spaces.Box):
            obs_shape = env.observation_space.shape
            self.observation_space = gym.spaces.Box(low = 0, high = env.max_dist, shape=(obs_shape[0], ))
        else:
            raise TypeError(f"Unknown observation space type {type(self.observation_space)}. Expected gym.spaces.Box or Dict")
        self.norm = max(env.window_size)

    def observation(self, observation):
        if isinstance(self.observation_space, gym.spaces.Dict):
            return OrderedDict({
                "observation" : np.linalg.norm(observation["observation"], axis=1) / self.norm,
                "achieved_goal" : observation["achieved_goal"],
                "desired_goal" : observation["desired_goal"],
            })
        return np.linalg.norm(observation, axis=1) / self.norm
