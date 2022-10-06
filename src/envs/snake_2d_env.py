##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-10-05 3:40:00 pm
 # @copyright https://mit-license.org/
 #
# Gym specific imports
import gym
from gym import spaces
import pygame
# project imports
from snake import SnakeAI
from line import Line, intersection_with_obstacles
from utils import Direction, get_opposite_direction
# IO imports
import logging
from typing import List, Dict
from pathlib import Path
# DL import
import numpy as np
import torch

font_path = Path(__file__).parents[1].joinpath('graphics', 'arial.ttf').resolve(strict=True)
FONT = pygame.font.Font(font_path, 25)

PIXEL_SIZE = 20
BODY_PIXEL_SIZE = 12
OBSTACLE_SIZE_RANGE = (1, 3)
# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN = (0,255,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
GREY = (150, 150, 150)

REWARD_FOOD = 10
REWARD_COLISION = -10
REWARD_COLISION_FREE = -1

class Snake2dEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, width : int = 20, height : int = 20, size : int =5, nb_obs : int = 7, nb_obstacles : int = 0):
        assert width > 4 and height > 4, "Environment needs at least 5 pixels in each dimension"
        self.width = width
        self.height = height

        self.nb_obs = nb_obs
        assert nb_obstacles >= 0
        self.nb_obstacles = nb_obstacles

        self.size = size  # The size of the square grid
        self.window_size = (width * PIXEL_SIZE, height * PIXEL_SIZE)  # The size of the PyGame window
        self.max_dist = np.sqrt(self.window_size[0] **2 + self.window_size[1] **2 )

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = gym.spaces.Dict(
            {
                "snake_obs": spaces.Box(low = np.zeros((self.nb_obs, 2)), high = np.repeat([self.window_size], self.nb_obs, axis=0), shape=(self.nb_obs, 2), dtype=float),
                "food": spaces.Box(low = np.zeros(2), high = np.array([self.window_size]), dtype=float),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left"
        self.action_space = spaces.Discrete(3)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: torch.tensor([1, 0, 0]),
            1: torch.tensor([0, 1, 0]),
            2: torch.tensor([0, 0, 1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
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

    def _get_obs(self):
        self._agent_location = np.array(self.snake.head.center)
        self._target_location = np.array(self.food.center)
        return {"snake_obs": self._agent_location, "food": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initialise snake
        x = np.random.randint(2, (self.width - 2)) * PIXEL_SIZE
        y = np.random.randint(2, (self.height - 2)) * PIXEL_SIZE
        self.snake = SnakeAI(x, y, pixel_size=PIXEL_SIZE, body_pixel=BODY_PIXEL_SIZE)

        # Initialise score
        self.score = 0
        self.obstacles = []

        if self.nb_obstacles > 0:
            self.obstacles = self._populate_grid_with_obstacles()
        self._place_food()
        self._compute_collision_lines()
        self._iteration = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    # TODO : implementer la fonction step
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        self.snake.move_from_action(action)
        # direction = self._action_to_direction[action]
        self._agent_location = self.snake.head.center
        # An episode is done iff the snake has reached the food
        terminated = self.snake.head.colliderect(self.food)
        # Give a reward according to the condition
        if terminated:
            reward = REWARD_FOOD
        elif self._is_collision():
            reward = REWARD_COLISION
        else:
            reward = REWARD_COLISION_FREE

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))

        # Draw snake
        self.snake.draw(canvas)
        # Draw collision lines
        for direction, line in self.collision_lines.items():
            if line is not None:
                logging.debug(f'Drawing line {line} for direction {direction}')
                line.draw(self.display, GREY, BLUE1)
        # Draw food
        pygame.draw.rect(canvas, GREEN, self.food)
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.display, RED, obstacle)
        # Print the score
        text = FONT.render(f"Score: {self.score}", True, WHITE)
        canvas.blit(text, [0, 0])


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
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
        x = np.random.randint(0, self.width) * PIXEL_SIZE
        y = np.random.randint(0, self.height) * PIXEL_SIZE
        self.food = pygame.Rect(x, y, PIXEL_SIZE, PIXEL_SIZE)
        # Check to not place food inside the snake
        if self.food.collidelist(self.snake.body) != -1 or self.food.colliderect(self.snake.head):
            self._place_food()
        # Check to place the food at least 1 pixel away from a wall
        if self.food.inflate(2*PIXEL_SIZE, 2*PIXEL_SIZE).collidelist(self.obstacles) != -1:
            self._place_food()

    def _populate_grid_with_obstacles(self) -> List[pygame.Rect]:
        obstacles = []
        for _ in range(self.nb_obstacles):
            size = np.random.randint(*OBSTACLE_SIZE_RANGE) * PIXEL_SIZE
            obstacle = self._place_obstacle(size)
            obstacles.append(obstacle)
        return obstacles

    def _place_obstacle(self, size : int) -> pygame.Rect:
        x = np.random.randint(0, self.width) * size
        y = np.random.randint(0, self.height) * size

        obstacle = pygame.Rect(x, y, size, size)
        # check colision with the initial snake bounding box
        bounding_box_factor = 2 * (len(self.snake) - 1) * PIXEL_SIZE

        if obstacle.colliderect(self.snake.head.inflate(bounding_box_factor, bounding_box_factor)):
            obstacle = self._place_obstacle(size)

        # check inclusion inside other obstacles
        for i in obstacle.collidelistall(self.obstacles):
            # If the new obstacle is completely contained in an existing obstacle, reprocess
            if self.obstacles[i].contains(obstacle):
                obstacle = self._place_obstacle(size)
        return obstacle

    def _is_outside(self, rect : pygame.Rect = None):
        if rect is None:
            rect = self.snake.head
        return rect.x < 0 or rect.x + rect.width > self.width or rect.y < 0 or rect.y + rect.height > self.height

    def _is_collision(self):
        return self._is_outside() or self.snake.collide_with_itself() or self.snake.collide_with_obstacles(self.obstacles)


    def _init_collision_lines(self):
        playground = pygame.Rect((0,0), self.window_size)

        for direction in Direction:
            end_line = np.array(self.snake.head.center) + direction.value * self.max_dist
            line = Line(self.snake.head.center, tuple(end_line))
            start, end = playground.clipline(line.start, line.end)
            self.collision_lines[direction] = Line(start, end)

        # snake_head = self.snake.head
        # self.collision_lines[Direction.UP] = Line(snake_head.center, (snake_head.centerx, 0))
        # self.collision_lines[Direction.UP_RIGHT] = Line(snake_head.center, (snake_head.centerx + snake_head.centery, 0))
        # self.collision_lines[Direction.RIGHT] = Line(snake_head.center, (self.width, snake_head.centery))
        # self.collision_lines[Direction.DOWN_RIGHT] = Line(snake_head.center, (self.width, (self.width - snake_head.centerx) + snake_head.centery))
        # self.collision_lines[Direction.DOWN] = Line(snake_head.center, (snake_head.centerx, self.height))
        # self.collision_lines[Direction.DOWN_LEFT] = Line(snake_head.center, (snake_head.centerx - (self.height - snake_head.centery), self.height))
        # self.collision_lines[Direction.LEFT] = Line(snake_head.center, (0, snake_head.centery))
        # self.collision_lines[Direction.UP_LEFT] = Line(snake_head.center, (0, snake_head.centery - snake_head.centerx))

    def _get_collision_box(self, direction : Direction) -> pygame.Rect:
        snake_head = self.snake.head
        # define quantities to compute snake position
        right_dist = self.window_size[0] - snake_head.right
        bottom_dist = self.window_size[1] - snake_head.bottom

        if direction == Direction.UP:
            return pygame.Rect(snake_head.left, 0, snake_head.width, snake_head.top)
        if direction == Direction.UP_RIGHT:
            return pygame.Rect(snake_head.right, 0, right_dist, snake_head.centery)
            return pygame.Rect(snake_head.right, 0, self.width * PIXEL_SIZE - snake_head.right, snake_head.top)
        if direction == Direction.RIGHT:
            return pygame.Rect(snake_head.topright, (right_dist, snake_head.height))
        if direction == Direction.DOWN_RIGHT:
            return pygame.Rect(snake_head.bottomright, (right_dist, bottom_dist))
            return pygame.Rect(snake_head.bottomright, (self.width * PIXEL_SIZE - snake_head.right, self.height - snake_head.bottom))
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
        logging.debug(f"collision box for direction {direction} is : {collision_box} ")
        return collision_box

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
            collision_list = [self.obstacles[index] for index in collision_box.collidelistall(self.obstacles)]
            if collision_list:
                self.collision_lines[direction] = intersection_with_obstacles(self.collision_lines[direction], collision_list)

    # def _get_collision_line(self, initial_line : Line, collision_list : List[pygame.Rect]) -> Line:
    #     intersection_point = None
    #     # Check all possible collisions within the collision box
    #     intersection_point = initial_line.intersect_obstacles(collision_list)
    #     if intersection_point:
    #         return Line(initial_line.start, intersection_point)
    #     logging.debug(f'No intersection found between line {initial_line} and obstacles')
    #     return initial_line

    # def _update_collision_line(self, direction : Direction):
    #     collision_box = self._get_collision_box(direction)
    #     collision_line = self.collision_lines[direction]
    #     collision_list = [self.obstacles[index] for index in collision_box.collidelistall(self.obstacles)]
    #     intersection_point = None
    #     if collision_list:
    #         # Check all possible collisions within the collision box
    #         intersection_point = collision_line.intersect_obstacles(collision_list)
    #         if intersection_point:
    #             self.collision_lines[direction] = Line(collision_line.start, intersection_point)
    #     # Case where no obstacles are found in the bounding box or none of the available obstacle collide with the collision line
    #     if (not collision_list) or (not intersection_point):
    #         logging.debug(f'collision list is empty or no intersection point found in bounding box')

    #         collision = collision_line.intersect(collision_box)
    #         logging.debug(f'collision between line {collision_line} from head {self.snake.head} and bounding box {collision_box} is : {collision}')
    #         self.collision_lines[direction] = Line(*collision)

    # def _update_collision_lines(self):
    #     opposite_direction = get_opposite_direction(self.snake.direction)
    #     # multithreading for this task do not improve performances !
    #     for direction in Direction:
    #         # Check all particular cases before calling the update fonction
    #         if direction == opposite_direction:
    #             self.collision_lines[direction] = None
    #         elif (direction == Direction.UP) and (self.snake.head.top <= 0):
    #             self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.midtop)
    #         elif (direction == Direction.UP_RIGHT) and (self.snake.head.top <= 0 or self.snake.head.right >= self.width):
    #             self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.topright)
    #         elif (direction == Direction.RIGHT) and (self.snake.head.right >= self.width):
    #             self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.midright)
    #         elif (direction == Direction.DOWN_RIGHT) and (self.snake.head.bottom >= self.height or self.snake.head.right >= self.width):
    #             self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.bottomright)
    #         elif (direction == Direction.DOWN) and (self.snake.head.bottom >= self.height):
    #             self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.midbottom)
    #         elif (direction == Direction.DOWN_LEFT) and (self.snake.head.bottom >= self.height or self.snake.head.left <= 0):
    #             self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.bottomleft)
    #         elif (direction == Direction.LEFT) and (self.snake.head.left <= 0):
    #             self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.midleft)
    #         elif (direction == Direction.UP_LEFT) and (self.snake.head.left <= 0 or self.snake.head.top <= 0):
    #             self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.topleft)
    #         else:
    #             self._update_collision_line(direction)

