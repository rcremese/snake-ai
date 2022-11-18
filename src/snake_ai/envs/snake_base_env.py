##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-11-10 11:27:11 pm
 # @copyright https://mit-license.org/
 #
from abc import ABCMeta, abstractmethod
from snake_ai.utils.paths import FONT_PATH
from snake_ai.envs.snake import Snake, SnakeAI
from snake_ai.utils import Colors
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
        self._snake = None
        self.score = None
        self._food = None
        self.truncated = None
        self._obstacles = []

    ## Abstract methods definition
    @abstractmethod
    def reset(self) -> None:
        # Initialise snake
        x = np.random.randint(2, (self.width - 2)) * self._pixel_size
        y = np.random.randint(2, (self.height - 2)) * self._pixel_size
        self._snake = SnakeAI(x, y, pixel_size=self._pixel_size)

        # Initialise score and obstacles
        self.score = 0
        self._obstacles = []
        if self.nb_obstacles > 0:
            self._obstacles = self._populate_grid_with_obstacles()
        self._food = self._place_food()
        self.truncated = False
        return None

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def _get_obs(self):
        raise NotImplementedError

    ## Public methods definition
    def render(self, mode="human"):
        canvas = pygame.Surface(self.window_size)
        canvas.fill(Colors.BLACK.value)

        self.draw(canvas)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            # Draw the text showing the score
            text = self.font.render(f"Score: {self.score}", True, Colors.WHITE.value)
            self.window.blit(text, [0, 0])
            # update the display
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def draw(self, canvas : pygame.Surface):
        # Draw snake
        self._snake.draw(canvas)
        # Draw obstacles
        for obstacle in self._obstacles:
            pygame.draw.rect(canvas, Colors.RED.value, obstacle)
        # Draw food
        pygame.draw.rect(canvas, Colors.GREEN.value, self._food)

    def close(self):
        if self.render_mode == "human":
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed : int = 42):
        np.random.seed(seed)

    ## Properties definitions
    @property
    def info(self):
        "Informations about the snake game states."
        return {
            "snake_direction": self._snake.direction,
            "obstacles": self._obstacles,
            "snake_head": self._snake.head.center,
            "food": self._food.center,
            "truncated": self.truncated,
        }

    @property
    def snake(self):
        "Snake agent represented by a subclass of Snake."
        return self._snake

    @snake.setter
    def snake(self, snake : Snake):
        if not (isinstance(snake, Snake) or snake is None):
            raise TypeError(f"The snake argument need to be an instance of Snake, not {type(snake)}")
        if (snake.head.x % self._pixel_size != 0) or (snake.head.y % self._pixel_size != 0):
            raise ValueError(f"Snake position need to be a factor of pixel size {self._pixel_size}. Currently, the snake head position is ({snake.head.x}, {snake.head.y}) ")
        self._snake = snake

    @property
    def food(self):
        "Food represented by an instance of pygame.Rect."
        return self._food

    @food.setter
    def food(self, food : pygame.Rect):
        if not (isinstance(food, pygame.Rect) or food is None):
            raise TypeError(f"The food argument need to be an instance of pygame.Rect, not {type(food)}")
        if (food.x % self._pixel_size != 0) or (food.y % self._pixel_size != 0):
            raise ValueError(f"Food position need to be a factor of pixel size {self._pixel_size}, not ({food.x}, {food.y}) ")
        self._food = food

    @property
    def obstacles(self):
        "Obstacles in the environment represented by a list of pygame.Rect"
        return self._obstacles

    @obstacles.setter
    def obstacles(self, obstacles : List[pygame.Rect]):
        if not ((isinstance(obstacles, list) and all([isinstance(elem, pygame.Rect) for elem in obstacles])) or (obstacles is None)):
            raise TypeError(f"The obstacles argument need to be an instance of List[pygame.Rect], not {type(obstacles)}")
        for idx, obstacle in enumerate(obstacles):
            if (obstacle.x % self._pixel_size != 0) or (obstacle.y % self._pixel_size != 0):
                raise ValueError(f"Obstacles positions need to be a factor of pixel size {self._pixel_size}. Obstacle at index {idx} has position ({obstacle.x}, {obstacle.y}) ")
        self._obstacles = obstacles

    ## Private methods definition
    def _place_food(self) -> pygame.Rect:
        # TODO : control the recursivity of the method
        # Define the central coordinates of the food to be placed
        x = np.random.randint(0, self.width) * self._pixel_size
        y = np.random.randint(0, self.height) * self._pixel_size
        food = pygame.Rect(x, y, self._pixel_size, self._pixel_size)
        # Check to not place food inside the snake
        if (food.collidelist(self._snake.body) != -1) or food.colliderect(self._snake.head):
            return self._place_food()
        # Check to place the food at least 1 pixel away from a wall
        if food.inflate(2*self._pixel_size, 2*self._pixel_size).collidelist(self._obstacles) != -1:
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
        bounding_box_factor = 2 * (len(self._snake) - 1) * self._pixel_size

        if obstacle.colliderect(self._snake.head.inflate(bounding_box_factor, bounding_box_factor)):
            obstacle = self._place_obstacle(size)

        # check inclusion inside other obstacles
        for i in obstacle.collidelistall(self._obstacles):
            # If the new obstacle is completely contained in an existing obstacle, reprocess
            if self._obstacles[i].contains(obstacle):
                obstacle = self._place_obstacle(size)
        return obstacle

    def _is_outside(self, rect: pygame.Rect = None) -> bool:
        if rect is None:
            rect = self._snake.head
        return rect.x < 0 or rect.x + rect.width > self.window_size[0] or rect.y < 0 or rect.y + rect.height > self.window_size[1]

    def _collide_with_obstacles(self, rect: pygame.Rect = None) -> bool:
        if rect is None:
            return self._snake.collide_with_obstacles(self._obstacles)
        return rect.collidelist(self._obstacles) != -1

    def _collide_with_snake_body(self, rect: pygame.Rect = None) -> bool:
        if rect is None:
            return self._snake.collide_with_itself()
        return rect.collidelist(self._snake.body) != -1

    def _is_collision(self, rect : pygame.Rect = None) -> bool:
        return self._is_outside(rect) or self._collide_with_snake_body(rect) or self._collide_with_obstacles(rect)

    def __repr__(self) -> str:
        return f"{__class__.__name__}({self.render_mode!r}, {self.width!r}, {self.height!r}, {self.nb_obstacles!r}, {self._pixel_size!r}, {self._max_obs_size!r})"
