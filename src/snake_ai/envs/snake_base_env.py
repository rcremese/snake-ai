##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-11-10 11:27:11 pm
 # @copyright https://mit-license.org/
 #
from abc import ABCMeta, abstractmethod
from snake_ai.utils.paths import FONT_PATH
from snake_ai.envs.snake import SnakeAI
from typing import List
import numpy as np
import pygame
import gym

class SnakeBaseEnv(gym.Env, metaclass=ABCMeta):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode=None, width: int = 20, height: int = 20, nb_obstacles: int = 0, pixel: int = 20, max_obs_size : int = 3):
        super().__init__()
        if width < 5 or height < 5:
            raise ValueError(f"Environment needs at least 5 pixels in each dimension. Get ({width}, {height})")
        self.width = width
        self.height = height

        if nb_obstacles < 0:
            raise ValueError("Can not implement environment with negative number of obstacles")
        self.nb_obstacles = nb_obstacles

        if max_obs_size < 0:
            raise ValueError("Can not implement environment with a negative obstacle size")
        self._max_obs_size = max_obs_size

        if pixel < 0:
            raise ValueError("Can not implement environment with negative pixel size")
        self._pixel_size = pixel
        # window to define the canvas
        self.window_size = (width * self._pixel_size, height * self._pixel_size)
        self.max_dist = np.linalg.norm(self.window_size)

        if (render_mode is not None) and (render_mode not in self.metadata["render_modes"]):
            raise ValueError(f"render_mode should be eather None or in {self.metadata['render_modes']}. Got {render_mode}")
        self.render_mode = render_mode

        if render_mode == "human":
            pygame.init()
            # instanciation of arguments that will be used by pygame when drawing
            self.window = pygame.display.set_mode(self.window_size)
            self.font = pygame.font.Font(FONT_PATH, 25)
            self.clock = pygame.time.Clock()

        # Set all existing attibutes to None
        self.observation_space = None
        self.action_space = None
        self.snake = None
        self.score = None
        self.food = None
        self.truncated = None
        self.obstacles = []

    def __repr__(self) -> str:
        return f"{__class__.__name__}({self.render_mode!r}, {self.width!r}, {self.height!r}, {self.nb_obstacles!r}, {self._pixel_size!r}, {self._max_obs_size!r})"

    @abstractmethod
    def reset(self):
        # Initialise snake
        x = np.random.randint(2, (self.width - 2)) * self._pixel_size
        y = np.random.randint(2, (self.height - 2)) * self._pixel_size
        self.snake = SnakeAI(x, y, pixel_size=self._pixel_size)

        # Initialise score and obstacles
        self.score = 0
        self.obstacles = []
        if self.nb_obstacles > 0:
            self.obstacles = self._populate_grid_with_obstacles()
        self.food = self._place_food()
        self.truncated = False

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def render(self, mode="human"):
        raise NotImplementedError

    @abstractmethod
    def _get_obs(self):
        raise NotImplementedError

    @property
    def info(self):
        "Informations about the snake game states."
        return {
            "snake_direction": self.snake.direction,
            "obstacles": self.obstacles,
            "snake_head": self.snake.head.center,
            "food": self.food.center,
            "truncated": self.truncated,
        }

    def close(self):
        if self.render_mode == "human":
            pygame.display.quit()
            pygame.quit()

    def _place_food(self) -> pygame.Rect:
        # TODO : control the recursivity of the method
        # Define the central coordinates of the food to be placed
        x = np.random.randint(0, self.width) * self._pixel_size
        y = np.random.randint(0, self.height) * self._pixel_size
        food = pygame.Rect(x, y, self._pixel_size, self._pixel_size)
        # Check to not place food inside the snake
        if (food.collidelist(self.snake.body) != -1) or food.colliderect(self.snake.head):
            return self._place_food()
        # Check to place the food at least 1 pixel away from a wall
        if food.inflate(2*self._pixel_size, 2*self._pixel_size).collidelist(self.obstacles) != -1:
            return self._place_food()
        # If everything works, return the food
        return food

    def _populate_grid_with_obstacles(self) -> List[pygame.Rect]:
        obstacles = []
        for _ in range(self.nb_obstacles):
            size = np.random.randint(1, self._max_obs_size)
            obstacle = self._place_obstacle(size)
            obstacles.append(obstacle)
        return obstacles

    def _place_obstacle(self, size: int) -> pygame.Rect:
        # TODO : control the recursivity of the method
        x = np.random.randint(0, self.width) * self._pixel_size
        y = np.random.randint(0, self.height) * self._pixel_size

        obstacle = pygame.Rect(x, y, size * self._pixel_size, size * self._pixel_size)
        # check colision with the initial snake bounding box
        bounding_box_factor = 2 * (len(self.snake) - 1) * self._pixel_size

        if obstacle.colliderect(self.snake.head.inflate(bounding_box_factor, bounding_box_factor)):
            obstacle = self._place_obstacle(size)

        # check inclusion inside other obstacles
        for i in obstacle.collidelistall(self.obstacles):
            # If the new obstacle is completely contained in an existing obstacle, reprocess
            if self.obstacles[i].contains(obstacle):
                obstacle = self._place_obstacle(size)
        return obstacle

    def _is_outside(self, rect: pygame.Rect = None) -> bool:
        if rect is None:
            rect = self.snake.head
        return rect.x < 0 or rect.x + rect.width > self.window_size[0] or rect.y < 0 or rect.y + rect.height > self.window_size[1]

    def _collide_with_obstacles(self, rect: pygame.Rect = None) -> bool:
        if rect is None:
            return self.snake.collide_with_obstacles(self.obstacles)
        return rect.collidelist(self.obstacles) != -1

    def _collide_with_snake_body(self, rect: pygame.Rect = None) -> bool:
        if rect is None:
            return self.snake.collide_with_itself()
        return rect.collidelist(self.snake.body) != -1

    def _is_collision(self, rect : pygame.Rect = None) -> bool:
        return self._is_outside(rect) or self._collide_with_snake_body(rect) or self._collide_with_obstacles(rect)

    def seed(self, seed : int = 42):
        np.random.seed(seed)
