##
# @author Robin CREMESE <robin.cremese@gmail.com>
# @file Description
# @desc Created on 2022-11-10 11:27:11 pm
# @copyright https://mit-license.org/
#
from abc import ABCMeta, abstractmethod
from snake_ai.utils.paths import FONT_PATH
from snake_ai.utils.errors import CollisionError
from snake_ai.envs.snake import Snake
from snake_ai.envs.geometry import Geometry, Rectangle, Circle
from snake_ai.utils import Colors
from pathlib import Path
from typing import List, Union, Tuple, Optional
import numpy as np
import pygame
import json
import gym


class SnakeBaseEnv(gym.Env, metaclass=ABCMeta):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, width: int = 20, height: int = 20, nb_obstacles: int = 0, pixel: int = 20, max_obs_size: int = 1, seed: int = 0):
        super().__init__()
        if width < 5 or height < 5:
            raise ValueError(
                f"Environment needs at least 5 pixels in each dimension. Get ({width}, {height})")
        self.width = width
        self.height = height

        if nb_obstacles < 0:
            raise ValueError(
                "Can not implement environment with negative number of obstacles")
        self.nb_obstacles = nb_obstacles

        if max_obs_size <= 0:
            raise ValueError("Can not implement environment with a negative obstacle size")
        self._max_obs_size = max_obs_size
        # Construct a list with the number of obstacle samples per size (first value = 1 pixel).
        # If the number of obstacles can not be devided by the maximum value, samples are added to the obstacles of size 1
        self._nb_samples_per_size = self._max_obs_size * [self.nb_obstacles // self._max_obs_size, ]
        self._nb_samples_per_size[0] += self.nb_obstacles - sum(self._nb_samples_per_size)
        # Check if the area take by the obstacles, the food + and the snake is greater than the total area
        if sum([(size + 1)**2 * nb_obs for size, nb_obs in enumerate(self._nb_samples_per_size)]) + 4 > self.width * self.height:
            raise ValueError(f"There are too much obstacles ({self.nb_obstacles}) or with a range which is too wide {self._max_obs_size} for the environment ({self.width},{self.height}).")

        if pixel < 0:
            raise ValueError(
                "Can not implement environment with negative pixel size")
        self._pixel_size = pixel
        # window to define the canvas
        self.window_size = (width * self._pixel_size,
                            height * self._pixel_size)
        self.max_dist = np.linalg.norm(self.window_size)

        if (render_mode is not None) and (render_mode not in self.metadata["render_modes"]):
            raise ValueError(
                f"render_mode should be eather None or in {self.metadata['render_modes']}. Got {render_mode}")
        self.render_mode = render_mode

        if not isinstance(seed, int):
            raise TypeError(f"Seed need to be an int, not a {type(seed)}")
        self._seed = seed
        self.seed()

        # Set all existing attibutes to None
        self.observation_space = None
        self.action_space = None
        self._snake = None
        self.score = None
        self._food = None
        self.truncated = None
        self._free_positions = None
        self._obstacles = None
        self._window = None
        self._font = None
        self._clock = None

        if render_mode == "human":
            self._init_human_renderer()

    def _init_human_renderer(self):
        pygame.init()
        # instanciation of arguments that will be used by pygame when drawing
        self._window = pygame.display.set_mode(self.window_size)
        self._font = pygame.font.Font(FONT_PATH, 25)
        self._clock = pygame.time.Clock()

    # Abstract methods definition
    @abstractmethod
    def reset(self) -> None:
        # Initialise a grid of free positions and score
        self.score = 0
        self.truncated = False
        self._free_positions = np.ones((self.width, self.height))
        # Initialise obstacles
        self._obstacles = []
        if self.nb_obstacles > 0:
            self._obstacles = self._populate_grid_with_obstacles()
        # Initialise snake and food
        self._snake = self._place_snake()
        self._food = self._place_food()
        # Update the rng for the next reset
        self._seed += 1
        self.seed()

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def _get_obs(self):
        raise NotImplementedError

    def write(self, output_path: Union[Path, str], detailed: bool = False):
        """Write the environment in a json file

        Args:
            output_path (Path or str): Path to the output file. If the filename does not contain a suffix, append a .json to it.
            detailed (bool): Flag to choose if a detailed version of the environment is needed (food, snake and obstacles positions)

        Raises:
            TypeError: If the suffix of the input path exists and is not .json
        """
        output_path = Path(output_path).resolve()
        if output_path.suffix == '':
            output_path = output_path.with_suffix('.json')
        elif output_path.suffix != '.json':
            raise TypeError(
                f"The output file format is expected to be .json, not {output_path.suffix}")

        dictionary = {
            "width": self.width,
            "height": self.height,
            "pixel": self._pixel_size,
            "seed": self._seed,
            "nb_obstacles": self.nb_obstacles,
            "max_obs_size": self._max_obs_size,
            "render_mode": "None" if self.render_mode is None else self.render_mode,
        }
        if detailed:
            # Need to reset the environment if the elements are not instantiated
            if (self._food is None) or (self._snake is None) or (self._obstacles is None):
                self.reset()
            # Add the snake position to the dict
            dictionary["snake"] = self._snake.to_dict()
            # Add the food position to the dict
            if isinstance(self._food, Circle):
                dictionary["food"] = {"circle": self._food.to_dict()}
            elif isinstance(self._food, Rectangle):
                dictionary["food"] = {"rectangle": self._food.to_dict()}
            # Add the obstacles positions to the dict
            obstacles_list = []
            for obs in self._obstacles:
                if isinstance(obs, Circle):
                    obstacles_list.append({"circle": obs.to_dict()})
                elif isinstance(obs, Rectangle):
                    obstacles_list.append({'rectangle': obs.to_dict()})
            dictionary['obstacles'] = obstacles_list
        # Write the json file
        with open(output_path, 'x') as file:
            json.dump(dictionary, file)

    @classmethod
    def load(cls,  filepath: Union[str, Path]):
        filepath = Path(filepath).resolve(strict=True)
        with open(filepath, 'r') as file:
            env_dict: dict = json.load(file)
        keys = {'render_mode', 'width', 'height',
                'nb_obstacles', 'pixel', 'max_obs_size', 'seed'}
        assert keys.issubset(env_dict.keys()), f"One of the following keys is not in the input .json file {filepath.name} : {keys}"
        if env_dict['render_mode'] == 'None':
            env_dict['render_mode'] = None
        env = cls(env_dict['render_mode'], env_dict["width"], env_dict["height"],
                  env_dict["nb_obstacles"], env_dict["pixel"], env_dict["max_obs_size"], env_dict["seed"])
        # Check for additional informations in the dictionnary
        if 'snake' in env_dict:
            env.snake = Snake.from_dict(env_dict['snake'])
        if 'food' in env_dict:
            if "circle" in env_dict["food"]:
                env.food = Circle.from_dict(env_dict["food"]["circle"])
            elif "rectangle" in env_dict["food"]:
                env.food = Rectangle.from_dict(env_dict["food"]["rectangle"])
            else:
                raise KeyError(f"Unknown keys {env_dict['food'].keys()} for the food entry. Expected 'circle' or 'rectangle'.")
        if 'obstacles' in env_dict:
            assert len(env_dict["obstacles"]) == env_dict["nb_obstacles"], f"The given number of obstacles does not correspond to the 'nb_obstacles' parameter. The json file is not considered."
            obstacles = []
            for obstacle in env_dict["obstacles"]:
                if "circle" in obstacle:
                    obstacles.append(Circle.from_dict(obstacle["circle"]))
                elif "rectangle" in obstacle:
                    obstacles.append(Rectangle.from_dict(obstacle["rectangle"]))
                else:
                    raise KeyError(f"Unknown keys {obstacle.keys()} for the obstacles entry. Expected 'circle' or 'rectangle'.")
            env.obstacles = obstacles
        return env
    # Public methods definition:

    def render(self, mode="human", canvas=None):
        if (mode == "human") and (self._window is None):
            self._init_human_renderer()
        if canvas is None:
            canvas = pygame.Surface(self.window_size)
            canvas.fill(Colors.BLACK.value)

        self.draw(canvas)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self._window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            # Draw the text showing the score
            text = self._font.render(
                f"Score: {self.score}", True, Colors.WHITE.value)
            self._window.blit(text, [0, 0])
            # update the display
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self._clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def draw(self, canvas: pygame.Surface):
        assert isinstance(
            canvas, pygame.Surface), f"pygame.Surface is expected for canvas, not {type(canvas)}"
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

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = self._seed
        self.rng = np.random.default_rng(seed)

    # Properties definitions
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
    def snake(self, snake: Snake):
        if not (isinstance(snake, Snake) or snake is None):
            raise TypeError(
                f"The snake argument need to be an instance of Snake, not {type(snake)}")
        if (snake.head.x % self._pixel_size != 0) or (snake.head.y % self._pixel_size != 0):
            raise ValueError(
                f"Snake position need to be a factor of pixel size {self._pixel_size}. Currently, the snake head position is ({snake.head.x}, {snake.head.y}) ")
        self._snake = snake

    @property
    def food(self):
        "Food represented by an instance of pygame.Rect."
        return self._food

    @food.setter
    def food(self, food: Rectangle):
        if not (isinstance(food, (Rectangle, Circle)) or food is None):
            raise TypeError(
                f"The food argument need to be an instance of Rectangle or Circle, not {type(food)}")
        if (food.x % self._pixel_size != 0) or (food.y % self._pixel_size != 0):
            raise ValueError(
                f"Food position need to be a factor of pixel size {self._pixel_size}, not ({food.x}, {food.y}) ")
        self._food = food

    def _get_free_position_mask(self) -> np.ndarray:
        free_positions = np.ones((self.width, self.height), dtype=bool)
        for obstacle in self._obstacles:
            x_obs , y_obs = obstacle.x // self.pixel_size, obstacle.y // self.pixel_size
            obs_size = obstacle.width // self.pixel_size
            free_positions[x_obs:x_obs+obs_size, y_obs:y_obs+obs_size] = False
        x_food, y_food = self._food.x // self.pixel_size, self._food.y // self.pixel_size
        free_positions[x_food, y_food] = False
        return free_positions

    @property
    def free_positions(self) -> List[Tuple[int]]:
        "The x and y coordinates of all the free positions in pixel values. Exclude the snake, the obstacles and the food."
        free_position_mask = self._get_free_position_mask()
        return [(x, y) for x in range(self.width) for y in range(self.height) if free_position_mask[x, y]]

    @property
    def obstacles(self) -> List[Geometry]:
        "Obstacles in the environment represented by a list of pygame.Rect"
        return self._obstacles

    @obstacles.setter
    def obstacles(self, obstacles: List[pygame.Rect]):
        if not ((isinstance(obstacles, list) and all([isinstance(elem, pygame.Rect) for elem in obstacles])) or (obstacles is None)):
            raise TypeError(
                f"The obstacles argument need to be an instance of List[pygame.Rect], not {type(obstacles)}")
        for idx, obstacle in enumerate(obstacles):
            if (obstacle.x % self._pixel_size != 0) or (obstacle.y % self._pixel_size != 0):
                raise ValueError(
                    f"Obstacles positions need to be a factor of pixel size {self._pixel_size}. Obstacle at index {idx} has position ({obstacle.x}, {obstacle.y}) ")
        self._obstacles = obstacles

    @property
    def pixel_size(self):
        "Size of a box representing a pixel in the game"
        return self._pixel_size

    # Private methods definition
    def _place_snake(self) -> Snake:
        # As the snake is initialised along
        available_positions = [(x, y) for x in range(2, self.width) for y in range(self.height) if all(self._free_positions[x-2:x+1, y])]
        assert len(available_positions) > 0, "There is no available positions for the snake in the current environment"
        x, y = self.rng.choice(available_positions)
        snake = Snake([(x, y), (x-1, y), (x-2, y)], pixel=self._pixel_size)
        self._free_positions[x-2:x+1] = False
        return snake

    def _place_food(self, is_circle : bool = False) -> Union[Rectangle, Circle]:
        # Define the central coordinates of the food to be placed
        available_positions = [(x, y) for x in range(self.width) for y in range(self.height) if self._free_positions[x, y]]
        assert len(available_positions) > 0, "There is no available positions for the food in the current environment"
        x, y = self.rng.choice(available_positions)
        food = Rectangle(x * self._pixel_size, y * self._pixel_size, self._pixel_size, self._pixel_size)
        self._free_positions[x,y] = False
        if is_circle:
            return food.to_circle()
        return food

    def check_overlaps(self):
        """Check overlaps between the snake, the food and the obstacles.

        Raises:
            CollisionError: error raised if one of the snake body part or food collide with the obstacles in the environment or with themself
        """
        # Check collisions for the snake
        for snake_part in self._snake:
            if snake_part.colliderect(self._food):
                raise CollisionError(
                    f"The snake part {snake_part} collides with the food {self._food}.")
            collision_idx = snake_part.collidelist(self._obstacles)
            if collision_idx != -1:
                raise CollisionError(
                    f"The snake part {snake_part} collides with the obstacle {self._obstacles[collision_idx]}.")
        # Check collisions for the food
        food_collision = self._food.collidelist(self._obstacles)
        if food_collision != -1:
            raise CollisionError(
                f"The food {self._food} collides with the obstacle {self._obstacles[food_collision]}.")

    def _populate_grid_with_obstacles(self) -> List[Geometry]:
        obstacles = []
        # Loop over the obstacle sizes
        for i, nb_obstacle in enumerate(self._nb_samples_per_size[::-1]):
            size = self._max_obs_size - i
            for _ in range(nb_obstacle):
                obstacles.append(self._place_obstacle(size))
        return obstacles

    def _place_obstacle(self, size: int, is_circle : bool = False) -> Geometry:
        assert size > 0, f"Obstacle size need to be at least 1. Get {size}"
        available_positions = [(x, y) for x in range(self.width-(size-1)) for y in range(self.height-(size-1)) if self._free_positions[x, y]]
        assert len(available_positions) > 0, f"There is no available position for an obstacle of size {size}"
        x, y = self.rng.choice(available_positions)
        obstacle = Rectangle(x * self._pixel_size, y * self._pixel_size, size * self._pixel_size, size * self._pixel_size)
        # Remove all possible
        if size > 1:
            self._free_positions[x:x+(size-1), y:y+(size-1)] = False
        else:
            self._free_positions[x, y] = False
        # Return the proper istance depending on the flag "is_circle"
        if is_circle:
            return obstacle.to_circle()
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

    def _is_collision(self, rect: pygame.Rect = None) -> bool:
        return self._is_outside(rect) or self._collide_with_snake_body(rect) or self._collide_with_obstacles(rect)

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, SnakeBaseEnv), f"Can not compare equality with an instance of {type(other)}. Expected type is SnakeBaseEnv"
        size_check = (self.height == other.height) and (self.width == other.width) and (self.pixel_size == other.pixel_size)
        snake_check = self.snake == other.snake
        food_check = self.food == other.food
        obstacles_check = self.obstacles == other.obstacles
        return size_check and snake_check and food_check and obstacles_check

    def __repr__(self) -> str:
        return f"{__class__.__name__}(render_mode={self.render_mode!r}, width={self.width!r}, height={self.height!r}, nb_obstacles={self.nb_obstacles!r}, pixel_size={self._pixel_size!r}, max_obs_size={self._max_obs_size!r})"
