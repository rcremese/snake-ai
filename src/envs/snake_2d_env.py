##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-10-05 3:40:00 pm
 # @copyright https://mit-license.org/
 #
import gym
from gym import spaces
import pygame
import numpy as np
from snake import Snake
import logging
from typing import List
from pathlib import Path
import torch

font_path = Path(__file__).parents[1].joinpath('graphics', 'arial.ttf').resolve(strict=True)
font = pygame.font.Font(font_path, 25)

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

class Snake2dEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, width : int = 20, height : int = 20, size : int =5, nb_obs : int = 7, nb_obstacles : int = 0):
        self.width = width
        self.height = height
        self.nb_obs = nb_obs
        assert nb_obstacles >= 0
        self.nb_obstacles = nb_obstacles

        self.size = size  # The size of the square grid
        self.window_size = (width * PIXEL_SIZE, height * PIXEL_SIZE)  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                # TODO : fixer les bornes des observables
                "snake_obs": spaces.Box(low = 0, high = size - 1, shape=(self.nb_obs, 2), dtype=float),
                "goal": spaces.Box(low = 0, high = size - 1, shape=(2,), dtype=float),
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
        return {"snake_obs": self._agent_location, "target": self._target_location}

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
        x = torch.randint(2, (self.width - 2), (1,)).item() * PIXEL_SIZE
        y = torch.randint(2, (self.height - 2), (1,)).item() * PIXEL_SIZE
        self.snake = Snake(x, y, pixel_size=PIXEL_SIZE, body_pixel=BODY_PIXEL_SIZE)

        # Initialise score
        self.score = 0
        if self.nb_obstacles > 0:
            self._populate_grid_with_obstacles()
        self._place_food()
        self.collision_lines = {}
        self.frame_iteration = 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
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
        text = font.render(f"Score: {self.score}", True, WHITE)
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
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _place_food(self):
        # Define the central coordinates of the food to be placed
        x = torch.randint(0, self.width, (1,)).item() * PIXEL_SIZE
        y = torch.randint(0, self.height, (1,)).item() * PIXEL_SIZE
        self.food = pygame.Rect(x, y, PIXEL_SIZE, PIXEL_SIZE)
        # Check to not place food inside the snake
        if self.food.collidelist(self.snake.body) != -1 or self.food.colliderect(self.snake.head):
            self._place_food()
        # Check to place the food at least 1 pixel away from a wall
        if self.food.inflate(2*PIXEL_SIZE, 2*PIXEL_SIZE).collidelist(self.obstacles) != -1:
            self._place_food()

    def _populate_grid_with_obstacles(self) -> List[pygame.Rect]:
        self.obstacles = []

        for _ in range(self.nb_obstacles):
            size = torch.randint(*OBSTACLE_SIZE_RANGE, (1,)).item() * PIXEL_SIZE
            obstacle = self._place_obstacle(size)
            # update area by adding the obstacle area minus the overlapping areas with other obstacles
            self.obstacles.append(obstacle)

    def _place_obstacle(self, size : int) -> pygame.Rect:
        x = torch.randint(0, self.width, (1,)).item() * size
        y = torch.randint(0, self.height, (1,)).item() * size

        obstacle = pygame.Rect(x, y, size, size)
        # check colision with the initial snake bounding box
        bounding_box_factor = 2 * len(self.snake) * PIXEL_SIZE

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


    def _init_collision_lines(self) -> Dict[Direction, Line]:
        snake_head = self.snake.head
        self.collision_lines[Direction.UP] = Line(snake_head.center, (snake_head.centerx, 0))
        self.collision_lines[Direction.UP_RIGHT] = Line(snake_head.center, (snake_head.centerx + snake_head.centery, 0))
        self.collision_lines[Direction.RIGHT] = Line(snake_head.center, (self.width, snake_head.centery))
        self.collision_lines[Direction.DOWN_RIGHT] = Line(snake_head.center, (self.width, (self.width - snake_head.centerx) + snake_head.centery))
        self.collision_lines[Direction.DOWN] = Line(snake_head.center, (snake_head.centerx, self.height))
        self.collision_lines[Direction.DOWN_LEFT] = Line(snake_head.center, (snake_head.centerx - (self.height - snake_head.centery), self.height))
        self.collision_lines[Direction.LEFT] = Line(snake_head.center, (0, snake_head.centery))
        self.collision_lines[Direction.UP_LEFT] = Line(snake_head.center, (0, snake_head.centery - snake_head.centerx))

    def _get_collision_box(self, direction : Direction):
        snake_head = self.snake.head
        if direction == Direction.UP:
            collision_box = pygame.Rect(snake_head.left, 0, snake_head.width, snake_head.top)
        elif direction == Direction.UP_RIGHT:
            collision_box = pygame.Rect(snake_head.right, 0, self.width - snake_head.right, snake_head.top)
        elif direction == Direction.RIGHT:
            collision_box = pygame.Rect(snake_head.topright, (self.width - snake_head.right, snake_head.height))
        elif direction == Direction.DOWN_RIGHT:
            collision_box = pygame.Rect(snake_head.bottomright, (self.width - snake_head.right, self.height - snake_head.bottom))
        elif direction == Direction.DOWN:
            collision_box = pygame.Rect(snake_head.bottomleft, (snake_head.width, self.height - snake_head.bottom))
        elif direction == Direction.DOWN_LEFT:
            collision_box = pygame.Rect(0, snake_head.bottom, snake_head.left, self.height - snake_head.bottom)
        elif direction == Direction.LEFT:
            collision_box = pygame.Rect(0, snake_head.top, snake_head.left, snake_head.height)
        elif direction == Direction.UP_LEFT:
            collision_box = pygame.Rect(0, 0, snake_head.left, snake_head.top)
        else:
            raise ValueError(f'Unknown direction {direction}')
        logging.debug(f"collision box for direction {direction} is : {collision_box} ")
        return collision_box

    def _update_collision_line(self, direction : Direction):
        collision_box = self._get_collision_box(direction)
        collision_line = self.collision_lines[direction]
        collision_list = [self.obstacles[index] for index in collision_box.collidelistall(self.obstacles)]
        intersection_point = None
        if collision_list:
            # Check all possible collisions within the collision box
            intersection_point = collision_line.intersect_obstacles(collision_list)
            if intersection_point:
                self.collision_lines[direction] = Line(collision_line.start, intersection_point)
        # Case where no obstacles are found in the bounding box or none of the available obstacle collide with the collision line
        if (not collision_list) or (not intersection_point):
            logging.debug(f'collision list is empty or no intersection point found in bounding box')

            collision = collision_line.intersect(collision_box)
            logging.debug(f'collision between line {collision_line} from head {self.snake.head} and bounding box {collision_box} is : {collision}')
            self.collision_lines[direction] = Line(*collision)

    def _update_collision_lines(self):
        opposite_direction = get_opposite_direction(self.snake.direction)
        # multithreading for this task do not improve performances !
        for direction in Direction:
            # Check all particular cases before calling the update fonction
            if direction == opposite_direction:
                self.collision_lines[direction] = None
            elif (direction == Direction.UP) and (self.snake.head.top <= 0):
                self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.midtop)
            elif (direction == Direction.UP_RIGHT) and (self.snake.head.top <= 0 or self.snake.head.right >= self.width):
                self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.topright)
            elif (direction == Direction.RIGHT) and (self.snake.head.right >= self.width):
                self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.midright)
            elif (direction == Direction.DOWN_RIGHT) and (self.snake.head.bottom >= self.height or self.snake.head.right >= self.width):
                self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.bottomright)
            elif (direction == Direction.DOWN) and (self.snake.head.bottom >= self.height):
                self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.midbottom)
            elif (direction == Direction.DOWN_LEFT) and (self.snake.head.bottom >= self.height or self.snake.head.left <= 0):
                self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.bottomleft)
            elif (direction == Direction.LEFT) and (self.snake.head.left <= 0):
                self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.midleft)
            elif (direction == Direction.UP_LEFT) and (self.snake.head.left <= 0 or self.snake.head.top <= 0):
                self.collision_lines[direction] = Line(self.snake.head.center, self.snake.head.topleft)
            else:
                self._update_collision_line(direction)

