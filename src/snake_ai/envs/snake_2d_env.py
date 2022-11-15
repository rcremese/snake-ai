##
# @author Robin CREMESE <robin.cremese@gmail.com>
# @file Description
# @desc Created on 2022-10-05 3:40:00 pm
# @copyright https://mit-license.org/
#
# Gym specific imports
import gym
import gym.spaces
from gym.utils.play import play
from gym.envs.registration import EnvSpec
import pygame
from pyparsing import Optional
# project imports
from snake_ai.envs.snake import BODY_PIXEL_SIZE, SnakeAI
from snake_ai.utils.line import Line, intersection_with_obstacles
from snake_ai.utils import Direction, Reward, get_opposite_direction
# IO imports
import logging
from typing import Iterable, List, Dict, Tuple
from pathlib import Path
# DL import
import numpy as np
import torch

FONT_PATH = Path(__file__).parents[1].joinpath(
    'graphics', 'arial.ttf').resolve(strict=True)

#TODO : remplacer ref. couleur par classe Colors !
BODY_PIXEL_SIZE = 12
MAX_OBSTACLE_SIZE = 3
TOLERANCE = 1 # number of pixels accepted for cliped line
# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 255, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREY = (150, 150, 150)

class Snake2dEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}
    spec = EnvSpec("Snake-v0")

    def __init__(self, render_mode=None, width: int = 20, height: int = 20, nb_obstacles: int = 0, pixel: int = 20):
        super().__init__()
        assert width > 4 and height > 4, "Environment needs at least 5 pixels in each dimension"
        self.width = width
        self.height = height
        assert nb_obstacles >= 0
        self.nb_obstacles = nb_obstacles
        self._nb_obs = len(Direction)
        self.collision_lines = {}
        self._pixel_size = pixel
        self.truncated = False
        # The size of the PyGame window
        self.window_size = (width * self._pixel_size, height * self._pixel_size)
        self.max_dist = np.sqrt(
            self.window_size[0] ** 2 + self.window_size[1] ** 2)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = gym.spaces.Box(low=np.zeros((self._nb_obs, 2)), high=np.repeat([self.window_size], self._nb_obs, axis=0), shape=(self._nb_obs, 2))
        self.action_space = gym.spaces.Discrete(3)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        if render_mode == "human":
            pygame.init()
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.font = None

    # provide correct observations
    def _get_obs(self):
        observables = np.zeros((self._nb_obs, 2))
        i = 0
        for direction in Direction:
            # only consider lines in front of the snake
            if self.collision_lines[direction] is None:
                continue
            observables[i, :] = self.collision_lines[direction].end
            i += 1
        # add the target to observation
        observables[-1, :] = self._food.center
        return observables

    @property
    def pixel_size(self):
        "Pixel size"
        return self._pixel_size

    @property
    def info(self):
        "Informations about the snake game states."
        return {
            "collision_lines": self.collision_lines,
            "snake_direction": self.snake.direction,
            "obstacles": self.obstacles,
            "snake_head": self.snake.head.center,
            "food": self._food.center,
            "truncated": self.truncated,
        }

    def reset(self):
        # Initialise snake
        x = np.random.randint(2, (self.width - 2)) * self._pixel_size
        y = np.random.randint(2, (self.height - 2)) * self._pixel_size
        self.snake = SnakeAI(x, y, pixel_size=self._pixel_size,
                             body_pixel=BODY_PIXEL_SIZE)

        # Initialise score
        self.score = 0
        self.obstacles = []

        if self.nb_obstacles > 0:
            self.obstacles = self._populate_grid_with_obstacles()
        self._place_food()
        self._compute_collision_lines()

        observation = self._get_obs()

        # if self.render_mode:
        #     self.render()
        return observation

    # TODO : mettre un peu d'ordre dans la fonction step
    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        # Map the action (element of {0,1,2}) to the direction we walk in
        self.snake.move_from_action(action)
        # Do not compute collision lines when the snake is out of bound
        if not self._is_outside():
            self._compute_collision_lines()

        # An episode is done iff the snake has reached the food
        self.truncated = self.snake.head.colliderect(self._food)
        # Give a reward according to the condition
        if self.truncated:
            reward = Reward.FOOD.value
            self.snake.grow()
            self.score += 1
            self._place_food()
        elif self._is_collision():
            reward = Reward.COLLISION.value
        else:
            # TODO: check the best reward
            reward = Reward.COLLISION_FREE.value
            # reward = np.exp(-np.linalg.norm(Line(self.snake.head.center, self._food.center).to_vector() / self.pixel_size))
        terminated = self._is_outside() or self.snake.collide_with_itself()
        observation = self._get_obs()

        # if (self.render_mode is not None) and not terminated:
        #     self.render()

        return observation, reward, terminated, self.info

    def render(self, mode):
        if mode == "human":
            if self.window is None:
                self.window = pygame.display.set_mode(self.window_size)
            if self.font is None:
                self.font = pygame.font.Font(FONT_PATH, 25)
            if self.clock is None:
                self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill(BLACK)

        # Draw snake
        self.snake.draw(canvas)
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(canvas, RED, obstacle)

        # Draw collision lines
        for direction, line in self.collision_lines.items():
            if line is not None:
                logging.debug(f'Drawing line {line} for direction {direction}')
                line.draw(canvas, GREY, BLUE1)
        # Draw food
        pygame.draw.rect(canvas, GREEN, self._food)
        # Draw line to food
        Line(self.snake.head.center, self._food.center).draw(canvas, GREEN, GREEN)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            # Draw the text showing the score
            text = self.font.render(f"Score: {self.score}", True, WHITE)
            self.window.blit(text, [0, 0])
            # update only the snake and the food
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _place_food(self):
        # Define the central coordinates of the food to be placed
        x = np.random.randint(0, self.width) * self._pixel_size
        y = np.random.randint(0, self.height) * self._pixel_size
        self._food = pygame.Rect(x, y, self._pixel_size, self._pixel_size)
        # Check to not place food inside the snake
        if self._food.collidelist(self.snake.body) != -1 or self._food.colliderect(self.snake.head):
            self._place_food()
        # Check to place the food at least 1 pixel away from a wall
        if self._food.inflate(2*self._pixel_size, 2*self._pixel_size).collidelist(self.obstacles) != -1:
            self._place_food()

    def _populate_grid_with_obstacles(self) -> List[pygame.Rect]:
        obstacles = []
        for _ in range(self.nb_obstacles):
            size = np.random.randint(1, MAX_OBSTACLE_SIZE)
            obstacle = self._place_obstacle(size)
            obstacles.append(obstacle)
        return obstacles

    def _place_obstacle(self, size: int) -> pygame.Rect:
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

    def _is_outside(self, rect: pygame.Rect = None):
        if rect is None:
            rect = self.snake.head
        return rect.x < 0 or rect.x + rect.width > self.window_size[0] or rect.y < 0 or rect.y + rect.height > self.window_size[1]

    def _collide_with_obstacles(self, rect: pygame.Rect = None):
        # if no rectangle is passed, check snake collision with obstacles
        if rect is None:
            return self.snake.collide_with_obstacles(self.obstacles)
        return rect.collidelist(self.obstacles) != -1

    def _collide_with_snake_body(self, rect: pygame.Rect = None):
        # if no rectangle is passed, check snake collision with obstacles
        if rect is None:
            return self.snake.collide_with_itself()
        return rect.collidelist(self.snake.body) != -1

    def _is_collision(self, rect : pygame.Rect = None):
        return self._is_outside(rect) or self._collide_with_snake_body(rect) or self._collide_with_obstacles(rect)

    def _init_collision_lines(self):
        playground = pygame.Rect((0, 0), self.window_size)

        for direction in Direction:
            end_line = np.array(self.snake.head.center) + \
                np.array(direction.value) * self.max_dist
            line = Line(self.snake.head.center, tuple(end_line))
            start, end = playground.clipline(line.start, line.end)
            # Deal with the limitations of pygame that
            # "The rect.bottom and rect.right attributes of a pygame.Rectpygame object for storing rectangular coordinates always lie one pixel outside of its actual border"
            if abs(end[0] - self.window_size[0]) <= TOLERANCE:
                end = (self.window_size[0], end[1])
            if end[0] <= TOLERANCE:
                end = (0, end[1])
            if end[1] <= TOLERANCE:
                end = (end[0], 0)
            if abs(end[1] - self.window_size[1]) <= TOLERANCE:
                end = (end[0], self.window_size[1])
            self.collision_lines[direction] = Line(start, end)

    def _get_collision_box(self, direction: Direction) -> pygame.Rect:
        snake_head = self.snake.head
        # define quantities to compute snake position
        right_dist = self.window_size[0] - snake_head.right
        bottom_dist = self.window_size[1] - snake_head.bottom

        if direction == Direction.UP:
            return pygame.Rect(snake_head.left, 0, snake_head.width, snake_head.top)
        if direction == Direction.UP_RIGHT:
            return pygame.Rect(snake_head.right, 0, right_dist, snake_head.centery)
        if direction == Direction.RIGHT:
            return pygame.Rect(snake_head.topright, (right_dist, snake_head.height))
        if direction == Direction.DOWN_RIGHT:
            return pygame.Rect(snake_head.bottomright, (right_dist, bottom_dist))
        if direction == Direction.DOWN:
            return pygame.Rect(snake_head.bottomleft, (snake_head.width, bottom_dist))
        if direction == Direction.DOWN_LEFT:
            return pygame.Rect(0, snake_head.bottom, snake_head.left, bottom_dist)
        if direction == Direction.LEFT:
            return pygame.Rect(0, snake_head.top, snake_head.left, snake_head.height)
        if direction == Direction.UP_LEFT:
            return pygame.Rect(0, 0, snake_head.left, snake_head.top)
        else:
            raise ValueError(f'Unknown direction {direction}')

    def _compute_collision_lines(self):
        # Initialise collision lines without obstacles
        self._init_collision_lines()
        snake_opposite_dir = get_opposite_direction(self.snake.direction)
        self.collision_lines[snake_opposite_dir] = None
        # Case where there is no obstacles to consider
        if self.nb_obstacles == 0:
            return None

        for direction in Direction:
            # Do not compute collision line for opposite direction of the snake
            if direction == snake_opposite_dir:
                continue
            collision_box = self._get_collision_box(direction)
            # Do not consider case where the snake is on the border
            if collision_box.width <= 0 or collision_box.height <= 0:
                continue
            # get all the obstacles that can collide with the collision line
            collision_list = [self.obstacles[index]
                              for index in collision_box.collidelistall(self.obstacles)]
            if collision_list:
                self.collision_lines[direction] = intersection_with_obstacles(
                    self.collision_lines[direction], collision_list)

    def seed(self, seed : int = 42):
        np.random.seed(seed)
        torch.random.manual_seed(seed)

if __name__ == '__main__':
    env = gym.make('Snake-v0', render_mode='rgb_array', nb_obstacles=10)

    logging.basicConfig(level=logging.INFO)
    game = play(env, keys_to_action={"q": 0, "z": 1, "d": 2})
