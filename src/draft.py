import torch
import numpy as np
import pygame
import gym
from gym.spaces import Discrete, Box
from snake_ai.envs.utils import Direction, get_opposite_direction
from snake_ai.envs.line import Line
from snake_ai.envs.snake_2d_env import Snake2dEnv

a= np.eye(10)
print(a.shape, type(a.shape))
for row in a:
    print(row)