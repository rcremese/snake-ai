##
# @author  <robin.cremese@gmail.com>
# @file Gym environment for the snake game
# @desc Created on 2023-03-20 1:17:23 pm
# @copyright MIT License
#
import gymnasium as gym
import pygame
from snake_ai.envs.snake import Snake, SnakeHuman, BidirectionalSnake
from snake_ai.envs.random_obstacles_env import RandomObstaclesEnv
from snake_ai.envs.grid_world import GridWorld
from snake_ai.utils import Reward, Direction
from snake_ai.utils.errors import ConfigurationError, InitialisationError
from typing import Optional, Tuple, Dict, List
import numpy as np


class SnakeEnv(RandomObstaclesEnv):
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        pixel: int = 20,
        nb_obs: int = 0,
        max_obs_size: int = 1,
        seed: int = 0,
        render_mode: Optional[str] = None,
        snake_type: str = "classic",
        **kwargs,
    ):
        super().__init__(width, height, pixel, nb_obs, max_obs_size, seed, render_mode)
        if not snake_type.lower() in ["classic", "bidirectional", "human"]:
            raise ValueError(
                f"Snake type must be either 'classic', 'bidirectional' or 'human'. Get {snake_type}"
            )
        self._snake_type = snake_type.lower()
        # Gym env attributes
        if self._snake_type == "classic":
            self.observation_space = gym.spaces.MultiBinary(11)
            self.action_space = gym.spaces.Discrete(3)
        else:
            self.observation_space = gym.spaces.MultiBinary(12)
            self.action_space = gym.spaces.Discrete(4)

    def reset(self, seed: Optional[int] = None):
        super().reset(seed)
        # Initialise the snake
        self.agent = self._place_snake()
        return self.observations, self.info

    def step(self, action: int) -> Tuple[np.ndarray, Reward, bool, Dict]:
        _, reward, terminated, info = super().step(action)
        # Check if the snake collided with the food
        if info["truncated"]:
            self.snake.grow()
        return self.observations, reward, terminated, info

    def draw(self, canvas: pygame.Surface):
        super().draw(canvas)
        self.agent.draw(canvas)

    ## Properties
    @GridWorld.name.getter
    def name(self) -> str:
        return f"SnakeEnv({self.width},{self.height})"

    @property
    def snake(self) -> Snake:
        "Snake agent represented by a subclass of Snake."
        return self.agent

    @GridWorld.observations.getter
    def observations(self):
        "Observation associated with the current state of the environment"
        if self._agent is None:
            raise InitialisationError(
                "The position is not initialised. Reset the environment first !"
            )
        neighbour_collisions = [
            self._is_collision(neighbour) for neighbour in self.snake.neighbours
        ]

        return np.array(
            [
                ## Neighbours collision
                *neighbour_collisions,
                ## Snake direction
                self.snake.direction == Direction.NORTH,  # UP
                self.snake.direction == Direction.EAST,  # RIGHT
                self.snake.direction == Direction.SOUTH,  # DOWN
                self.snake.direction == Direction.WEST,  # LEFT
                ## Food position
                self.goal.y < self.snake.position.y,  # UP
                self.goal.x > self.snake.position.x,  # RIGHT
                self.goal.y > self.snake.position.y,  # DOWN
                self.goal.x < self.snake.position.x,  # LEFT
            ],
            dtype=int,
        )

    # TODO : Think about a new way to initialize snake !
    def _place_snake(self) -> Snake:
        # As the snake is initialised along
        # available_positions = [(x, y) for x in range(2, self.width) for y in range(self.height) if all(self._free_positions[x-2:x+1, y])]
        # assert len(available_positions) > 0, "There is no available positions for the snake in the current environment"
        available_positions = self.free_positions
        self._rng.shuffle(available_positions)
        snake_positions = self._get_snake_positions(available_positions)
        # x, y = self.rng.choice(self.free_positions)
        # snake = SnakeAI(x * self.pixel, y * self.pixel, pixel=self.pixel)
        if self._snake_type == "classic":
            return Snake(snake_positions, pixel=self.pixel)
        elif self._snake_type == "bidirectional":
            return BidirectionalSnake(snake_positions, pixel=self.pixel)
        elif self._snake_type == "human":
            return SnakeHuman(snake_positions, pixel=self.pixel)
        raise ValueError(
            f"Unknown snake type {self._snake_type}. Expected 'classic', 'bidirectional' or 'human'."
        )

    def _get_snake_positions(
        self, available_positions: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        for x, y in available_positions:
            # RIGHT
            if (x + 1, y) in available_positions:
                if (x + 1, y - 1) in available_positions:
                    return [(x, y), (x + 1, y), (x + 1, y - 1)]
                if (x + 2, y) in available_positions:
                    return [(x, y), (x + 1, y), (x + 2, y)]
                if (x + 1, y + 1) in available_positions:
                    return [(x, y), (x + 1, y), (x + 1, y + 1)]
            # BOTTOM
            if (x, y + 1) in available_positions:
                if (x + 1, y + 1) in available_positions:
                    return [(x, y), (x, y + 1), (x + 1, y + 1)]
                if (x, y + 2) in available_positions:
                    return [(x, y), (x, y + 1), (x, y + 2)]
                if (x - 1, y + 1) in available_positions:
                    return [(x, y), (x, y + 1), (x - 1, y + 1)]
            # LEFT
            if (x - 1, y) in available_positions:
                if (x - 1, y + 1) in available_positions:
                    return [(x, y), (x - 1, y), (x - 1, y + 1)]
                if (x - 2, y) in available_positions:
                    return [(x, y), (x - 1, y), (x - 2, y)]
                if (x - 1, y - 1) in available_positions:
                    return [(x, y), (x - 1, y), (x - 1, y - 1)]
            # TOP
            if (x, y - 1) in available_positions:
                if (x - 1, y - 1) in available_positions:
                    return [(x, y), (x, y - 1), (x - 1, y - 1)]
                if (x, y - 2) in available_positions:
                    return [(x, y), (x, y - 1), (x, y - 2)]
                if (x + 1, y - 1) in available_positions:
                    return [(x, y), (x, y - 1), (x + 1, y - 1)]
        raise ConfigurationError(
            "There is no valid configuration in free space to place a 3 pixel snake."
        )

    # Collision handling
    def _collide_with_snake_body(self, rect: Optional[pygame.Rect] = None) -> bool:
        if rect is None:
            return self.snake.collide_with_itself()
        return rect.collidelist(self.snake.body) != -1

    def _is_collision(self, rect: Optional[pygame.Rect] = None) -> bool:
        if isinstance(self.agent, Snake):
            return super()._is_collision(rect) or self._collide_with_snake_body(rect)
        return super()._is_collision(rect)

    ## Dunder methods
    def __repr__(self) -> str:
        return (
            f"{__class__.__name__}(width={self.width!r}, height={self.height!r}, pixel={self.pixel!r}, nb_obstacles={self._nb_obs!r}, "
            + f"max_obs_size={self._max_obs_size!r}, render_mode={self.render_mode!r}, seed={self._seed}, snake_type={self._snake_type!r})"
        )


if __name__ == "__main__":
    snake_env = SnakeEnv(
        20, 20, nb_obs=10, max_obs_size=5, render_mode="human", snake_type="human"
    )
    seed = 0
    snake_env.reset(seed)

    action = 0
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                snake_env.close()
                quit()
            key_pressed = event.type == pygame.KEYDOWN
            if key_pressed and event.key == pygame.K_UP:
                action = 0
            if key_pressed and event.key == pygame.K_RIGHT:
                action = 1
            if key_pressed and event.key == pygame.K_DOWN:
                action = 2
            if key_pressed and event.key == pygame.K_LEFT:
                action = 3
        _, _, terminated, _ = snake_env.step(action)
        if terminated:
            seed += 1
            snake_env.reset(seed)
            print("You suck ! Try again !")
        snake_env.render()
