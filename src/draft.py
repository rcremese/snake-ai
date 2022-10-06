import torch
import numpy as np
import enum
import pygame
import gym
from gym.spaces import Discrete, Box

class Direction(enum.Enum):
    UP = (0,-1)
    UP_RIGHT = (1, -1)
    RIGHT = (1,0)
    DOWN_RIGHT = (1, 1)
    DOWN = (0,1)
    DOWN_LEFT = (-1, 1)
    LEFT = (-1, 0)
    UP_LEFT = (-1, -1)

class Foo():
    def __init__(self) -> None:
        self.x = 12
    def _get_x(self):
        return self.x

class Bar():
    def __init__(self) -> None:
        self.foo = Foo()
    def bar(self):
        self.foo._get_x()

bar = Bar()
print(bar.bar())
print(np.repeat([(1,10)], 2, axis=0))
box = Box(low=np.zeros((2,2)), high=np.repeat([[1,10]], 2, axis = 0))
for _ in range(10):
    print(box.sample())