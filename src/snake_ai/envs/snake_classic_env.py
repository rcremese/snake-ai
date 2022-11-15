from snake_ai.envs.snake_base_env import SnakeBaseEnv
from snake_ai.utils import Direction, Reward, Colors
from typing import List
import numpy as np
import pygame
import gym

class SnakeClassicEnv(SnakeBaseEnv):
    def __init__(self, render_mode=None, width: int = 20, height: int = 20, nb_obstacles: int = 0, pixel: int = 20, max_obs_size: int = 3):
        super().__init__(render_mode, width, height, nb_obstacles, pixel, max_obs_size)
        self.observation_space = gym.spaces.MultiBinary(11)
        self.action_space = gym.spaces.Discrete(3)

    def reset(self):
        super().reset()
        return self._get_obs()

    def step(self, action : int):
        self.snake.move_from_action(action)
        # A flag is set if the snake has reached the food
        self.truncated = self.snake.head.colliderect(self.food)
        # An other one if the snake is outside or collide with obstacles
        terminated = self._is_collision()
        # Give a reward according to the condition
        if self.truncated:
            reward = Reward.FOOD.value
            self.snake.grow()
            self.score += 1
            self._place_food()
        elif terminated:
            reward = Reward.COLLISION.value
        else:
            reward = Reward.COLLISION_FREE.value
            # reward = np.exp(-np.linalg.norm(Line(self.snake.head.center, self._food.center).to_vector() / self.pixel_size))
        return self._get_obs(), reward, terminated, self.info

    def render(self, mode="human"):
        canvas = pygame.Surface(self.window_size)
        canvas.fill(Colors.BLACK.value)

        # Draw snake
        self.snake.draw(canvas)
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(canvas, Colors.RED.value, obstacle)

        # Draw food
        pygame.draw.rect(canvas, Colors.GREEN.value, self.food)

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
            self.snake.direction == Direction.UP,
            # RIGHT
            self.snake.direction == Direction.RIGHT,
            # DOWN
            self.snake.direction == Direction.DOWN,
            # LEFT
            self.snake.direction == Direction.LEFT,

            ## Food position
            # UP
            self.food.y < self.snake.head.y,
            # RIGHT
            self.food.x > self.snake.head.x,
            # DOWN
            self.food.y > self.snake.head.y,
            # LEFT
            self.food.x < self.snake.head.x,
        ], dtype=int)

        return observations

    def _get_neighbours(self) -> List[pygame.Rect]:
        """Return left, front and right neighbouring bounding boxes

        Raises:
            ValueError: if the snake direction is not in [UP, RIGHT, DOWN, LEFT]

        Returns:
            _type_: _description_
        """
        bottom = self.snake.head.move(0, self._pixel_size)
        top = self.snake.head.move(0, -self._pixel_size)
        left = self.snake.head.move(-self._pixel_size, 0)
        right = self.snake.head.move(self._pixel_size, 0)

        if self.snake.direction == Direction.UP:
            return left, top, right
        if self.snake.direction == Direction.RIGHT:
            return top, right, bottom
        if self.snake.direction == Direction.DOWN:
            return right, bottom, left
        if self.snake.direction == Direction.LEFT:
            return bottom, left, top
        raise ValueError(f'Unknown direction {self.snake.direction}')