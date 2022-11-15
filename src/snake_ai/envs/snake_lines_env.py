##
# @author Robin CREMESE <robin.cremese@gmail.com>
# @file Description
# @desc Created on 2022-11-11 1:09:16 am
# @copyright https://mit-license.org/
#
from snake_ai.envs.snake_base_env import SnakeBaseEnv
from snake_ai.utils import Colors, Reward
from snake_ai.utils.direction import Direction, get_opposite_direction
from snake_ai.utils.line import Line, intersection_with_obstacles
import numpy as np
import pygame
import gym

TOLERANCE = 1  # number of pixels accepted for cliped line


class SnakeLinesEnv(SnakeBaseEnv):
    def __init__(self, render_mode=None, width: int = 20, height: int = 20, nb_obstacles: int = 0, pixel: int = 20, max_obs_size: int = 3):
        super().__init__(render_mode, width, height, nb_obstacles, pixel, max_obs_size)
        self._nb_obs = len(Direction)
        self.observation_space = gym.spaces.Box(low=np.zeros((self._nb_obs, 2)), high=np.repeat([self.window_size], self._nb_obs, axis=0),
                                                shape=(self._nb_obs, 2))
        self.action_space = gym.spaces.Discrete(3)
        self._food_line = None

    def reset(self):
        super().reset()
        self._compute_collision_lines()
        self._food_line = Line(self.snake.head.center, self.food.center)
        return self._get_obs()

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

    def step(self, action):
        # Map the action (element of {0,1,2}) to the direction we walk in
        self.snake.move_from_action(action)
        # Do not compute collision lines when the snake is out of bound
        if not self._is_outside():
            self._compute_collision_lines()
        self._food_line = Line(self.snake.head.center, self.food.center)

        # A flag is set if the snake has reached the food
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
            reward = Reward.COLLISION_FREE.value
            # reward = np.exp(-np.linalg.norm(Line(self.snake.head.center, self._food.center).to_vector() / self.pixel_size))
        terminated = self._is_outside() or self.snake.collide_with_itself()

        return self._get_obs(), reward, terminated, self.info

    def render(self, mode="human"):
        canvas = pygame.Surface(self.window_size)
        canvas.fill(Colors.BLACK.value)

        # Draw snake
        self.snake.draw(canvas)
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(canvas, Colors.RED.value, obstacle)

        # Draw collision lines
        for direction, line in self.collision_lines.items():
            if line is not None:
                line.draw(canvas, Colors.GREY.value, Colors.BLUE1.value)
        # Draw food
        pygame.draw.rect(canvas, Colors.GREEN.value, self.food)
        # Draw line to food
        self._food_line.draw(canvas, Colors.GREEN.value, Colors.GREEN.value)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            # Draw the text showing the score
            text = self.font.render(
                f"Score: {self.score}", True, Colors.WHITE.value)
            self.window.blit(text, [0, 0])
            # update the display
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def _init_collision_lines(self):
        playground = pygame.Rect((0, 0), self.window_size)

        for direction in Direction:
            end_line = np.array(self.snake.head.center) + \
                np.array(direction.value) * self.max_dist
            line = Line(self.snake.head.center, tuple(end_line))
            start, end = playground.clipline(line.start, line.end)
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
