import torch
import numpy as np
import pygame
import gym
from collections import OrderedDict
from gym.spaces import Discrete, Box, Dict
from snake_ai.envs.utils import Direction, get_opposite_direction
from snake_ai.envs.line import Line
from snake_ai.envs.snake_2d_env import Snake2dEnv

class A():
    def __init__(self) -> None:
        print("A")
        self.obs = Dict({'obs': Box(-1, 1, shape=(10,2)), 'test' : Box(-1, 1, shape=(2,))})

class B(A):
    def __init__(self) -> None:
        print("B")

a=A()
b=B()
c = OrderedDict({'toto' : 'tata', 'pipi' : 'caca'})
print(c["toto"])