##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-11-29 1:25:32 pm
 # @copyright https://mit-license.org/
 #
from snake_ai.envs.snake_base_env import SnakeBaseEnv
from snake_ai.envs.geometry import Circle, Rectangle
from snake_ai.utils import Direction, Reward, Colors
from typing import List, Tuple, Dict
import numpy as np
import pygame
import gym

class SnakeClassicEnv(SnakeBaseEnv):
    def __init__(self, render_mode=None, width: int = 20, height: int = 20, nb_obstacles: int = 0, pixel: int = 20, max_obs_size: int = 1, seed: int = 0):
        super().__init__(render_mode, width, height, nb_obstacles, pixel, max_obs_size, seed)
        self.observation_space = gym.spaces.MultiBinary(11)
        self.action_space = gym.spaces.Discrete(3)

    ## Abstract method definition
    def reset(self):
        super().reset()
        return self._get_obs()

    def step(self, action : int) -> Tuple[np.ndarray, Reward, bool, Dict]:
        self._snake.move_from_action(action)
        food = self._food.to_rectangle() if isinstance(self._food, Circle) else self._food
        self.truncated = self._snake.head.colliderect(food)
        # A flag is set if the snake has reached the food
        # An other one if the snake is outside or collide with obstacles
        terminated = self._is_collision()
        # Give a reward according to the condition
        if self.truncated:
            reward = Reward.FOOD.value
            self._snake.grow()
            self.score += 1
            self._food = self._place_food()
        elif terminated:
            reward = Reward.COLLISION.value
        else:
            reward = Reward.COLLISION_FREE.value
            # reward = np.exp(-np.linalg.norm(Line(self.snake.head.center, self._food.center).to_vector() / self.pixel_size))
        return self._get_obs(), reward, terminated, self.info

    ## Private method definition
    def _get_obs(self):
        left, front, right = self._get_neighbours()

        observations = np.array([
            ## Neighbours collision
            # LEFT
            self._is_collision(left),
            # FRONT
            self._is_collision(front),
            # RIGHT
            self._is_collision(right),

            ## Snake direction
            # UP
            self._snake.direction == Direction.UP,
            # RIGHT
            self._snake.direction == Direction.RIGHT,
            # DOWN
            self._snake.direction == Direction.DOWN,
            # LEFT
            self._snake.direction == Direction.LEFT,

            ## Food position
            # UP
            self._food.y < self._snake.head.y,
            # RIGHT
            self._food.x > self._snake.head.x,
            # DOWN
            self._food.y > self._snake.head.y,
            # LEFT
            self._food.x < self._snake.head.x,
        ], dtype=int)

        return observations

    def _get_neighbours(self) -> List[pygame.Rect]:
        """Return left, front and right neighbouring bounding boxes

        Raises:
            ValueError: if the snake direction is not in [UP, RIGHT, DOWN, LEFT]

        Returns:
            _type_: _description_
        """
        bottom = self._snake.head.move(0, self._pixel_size)
        top = self._snake.head.move(0, -self._pixel_size)
        left = self._snake.head.move(-self._pixel_size, 0)
        right = self._snake.head.move(self._pixel_size, 0)

        if self._snake.direction == Direction.UP:
            return left, top, right
        if self._snake.direction == Direction.RIGHT:
            return top, right, bottom
        if self._snake.direction == Direction.DOWN:
            return right, bottom, left
        if self._snake.direction == Direction.LEFT:
            return bottom, left, top
        raise ValueError(f'Unknown direction {self._snake.direction}')