##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-11-02 2:10:17 pm
 # @copyright https://mit-license.org/
 #

# TODO : Rename and rewrite the class to correspond to new case
import gym
import gym.spaces
import pygame
from snake_ai.envs import SnakeClassicEnv
import numpy as np
from typing import List
from snake_ai.utils import Direction, Colors
from snake_ai.utils.paths import FONT_PATH


class DiffusionWrapper(gym.Wrapper):
    def __init__(self, env: SnakeClassicEnv, diffusion_coef : float = 1):
        super().__init__(env)
        self.env : SnakeClassicEnv
        self.observation_space = gym.spaces.Box(low=np.zeros(3), high=np.ones(3), shape=(3,))
        self._diffusion_coef = diffusion_coef
        self._diffusive_field = None

    def reset(self):
        obs = self.env.reset()
        self._diffusive_field = self._get_diffusion_field()
        return self.observation(obs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if info['truncated']:
            self._diffusive_field = self._get_diffusion_field()
        return self.observation(obs), reward, done, info

    def observation(self, observation):
        assert self._diffusive_field is not None, "No diffusive field computed"
        observation = np.zeros(3)
        neighbours = self._get_neighbours()
        for i, neighbour in enumerate(neighbours):
            # observation is 0 if the neighbour bounding box collide with obtacles, snake body or is outside
            if self.env._is_collision(neighbour):
                continue
            window = [*neighbour.topleft, *neighbour.bottomright]
            observation[i] = np.mean(self._diffusive_field[window[0]:window[2], window[1]:window[3]])
        return observation

    def render(self, mode="human", **kwargs):
        if mode == "human":
            if self.window is None:
                self.window = pygame.display.set_mode(self.window_size)
            if self.clock is None:
                self.clock = pygame.time.Clock()
            if self.font is None:
                self.font = pygame.font.Font(FONT_PATH, 25)

        # fill canvas with the normalized diffusion field
        if self._diffusive_field is None:
            self._diffusive_field = self._get_diffusion_field()
        surf = np.zeros((*self.env.window_size, 3))
        surf[:,:,1] = 255 * self._diffusive_field # fill only the green part
        canvas = pygame.surfarray.make_surface(surf)

        # Draw snake
        self.env.snake.draw(canvas)
        # Draw obstacles
        for obstacle in self.env.obstacles:
            pygame.draw.rect(canvas, Colors.RED.value, obstacle)
        # Draw food
        # pygame.draw.lines(canvas, Colors.BLACK.value, closed=True, points=[self.env._food.topleft, self.env._food.topright, self.env._food.bottomright, self.env._food.bottomleft])
        pygame.draw.rect(canvas, Colors.BLACK.value, self.env._food, width=1)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump () # internally process pygame event handlers
            text = self.font.render(f"Score: {self.score}", True, Colors.WHITE.value)
            self.window.blit(text, [0, 0])

            pygame.display.update()
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def _get_diffusion_field(self) -> np.ndarray:
        x,y = np.meshgrid(range(self.env.window_size[0]), range(self.env.window_size[0]))
        mu = self.env._food.center
        diffusive_field = np.exp(- 0.5 / self._diffusion_coef *((x - mu[1])**2 + (y - mu[0])**2))
        assert np.max(diffusive_field) != np.min(diffusive_field), "Diffion field is constant"
        return (diffusive_field - np.min(diffusive_field)) / (np.max(diffusive_field) - np.min(diffusive_field))

    def _get_neighbours(self) -> List[pygame.Rect]:
        """Return left, front and right neighbouring bounding boxes

        Raises:
            ValueError: if the snake direction is not in [UP, RIGHT, DOWN, LEFT]

        Returns:
            _type_: _description_
        """
        bottom = self.env.snake.head.move(0, self.env.pixel_size)
        top = self.env.snake.head.move(0, -self.env.pixel_size)
        left = self.env.snake.head.move(-self.env.pixel_size, 0)
        right = self.env.snake.head.move(self.env.pixel_size, 0)

        if self.env.snake.direction == Direction.UP:
            return left, top, right
        if self.env.snake.direction == Direction.RIGHT:
            return top, right, bottom
        if self.env.snake.direction == Direction.DOWN:
            return right, bottom, left
        if self.env.snake.direction == Direction.LEFT:
            return bottom, left, top
        raise ValueError(f'Unknown direction {self.env.snake.direction}')