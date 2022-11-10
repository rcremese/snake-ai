import torch
import numpy as np
import pygame
import gym
from collections import OrderedDict
from gym.spaces import Discrete, Box, Dict
from snake_ai.envs.utils import Direction, get_opposite_direction
from snake_ai.envs.line import Line
from snake_ai.envs.snake_2d_env import Snake2dEnv
import matplotlib.pyplot as plt

obs = pygame.Rect(0, 0, 10, 10)
point = np.array([10,20])
print(obs.collidepoint(*point))

copy = np.copy(point)
copy[1] = 42
print(point, copy)