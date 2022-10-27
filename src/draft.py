import torch
import numpy as np
import pygame
import gym
from gym.spaces import Discrete, Box
from snake_ai.envs.utils import Direction, get_opposite_direction
from snake_ai.envs.line import Line
from snake_ai.envs.snake_2d_env import NB_OBS, Snake2dEnv

rect = pygame.Rect(0,0, 10, 10)
line = ((5, 5), (5, 15))
cliped_line = rect.clipline(line)
print(cliped_line)
