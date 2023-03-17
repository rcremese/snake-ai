import gym
import gym.spaces
import pygame
from snake_ai.envs.snake import Snake
from snake_ai.envs.random_obstacles_env import RandomObstaclesEnv
from snake_ai.envs.geometry import Rectangle
from snake_ai.utils import Colors, Reward, Direction
from snake_ai.utils.errors import CollisionError, ConfigurationError, InitialisationError
from typing import Optional, Tuple, Dict, List
import numpy as np

class SnakeEnv(RandomObstaclesEnv):
    def __init__(self, width: int = 20, height: int = 20, pixel: int = 20, nb_obs: int = 0,
                 max_obs_size: int = 1, seed: int = 0, render_mode : Optional[str]=None):
        super().__init__(width, height, pixel, nb_obs, max_obs_size, seed, render_mode)
        # Gym env attributes
        self.observation_space = gym.spaces.MultiBinary(11)
        self.action_space = gym.spaces.Discrete(3)
        # All non-instanciated attributes
        self._snake = None

    def reset(self, seed : Optional[int]=None):
        self.seed(seed)
        # Initialise a grid of free positions and score
        self.score = 0
        # self.truncated = False
        self._free_position_mask = np.ones((self.width, self.height))
        # Initialise obstacles
        self._obstacles = []
        if self.nb_obs > 0:
            self._obstacles = self._populate_grid_with_obstacles()
        # Initialise goal & position
        self._goal = self._place_goal()
        self._snake = self._place_snake()
        self._position = self._snake.head
        self._direction = self._snake.direction
        self._truncated = False
        return self.observations, self.info

    def step(self, action : int) -> Tuple[np.ndarray, Reward, bool, Dict]:
        self._snake.move_from_action(action)
        self._position = self._snake.head
        self._direction = self._snake.direction

        self._truncated = self._position.colliderect(self._goal)
        # A flag is set if the snake has reached the food
        # An other one if the snake is outside or collide with obstacles
        terminated = self._is_collision()
        # Give a reward according to the condition
        if self._truncated:
            reward = Reward.FOOD.value
            self._snake.grow()
            self.score += 1
            self._goal = self._place_goal()
        elif terminated:
            reward = Reward.COLLISION.value
        else:
            reward = Reward.COLLISION_FREE.value
        return self.observations, reward, terminated, self.info

    def draw(self, canvas: pygame.Surface):
        super().draw(canvas)
        self._snake.draw(canvas)

    # def check_overlaps(self):
    #     """Check overlaps between the snake, the food and the obstacles.

    #     Raises:
    #         CollisionError: error raised if one of the snake body part or food collide with the obstacles in the environment or with themself
    #     """
    #     # Check collisions for the snake
    #     for snake_part in self._snake:
    #         if snake_part.colliderect(self._goal):
    #             raise CollisionError(f"The snake part {snake_part} collides with the food {self._goal}.")
    #         collision_idx = snake_part.collidelist(self._obstacles)
    #         if collision_idx != -1:
    #             raise CollisionError(f"The snake part {snake_part} collides with the obstacle {self._obstacles[collision_idx]}.")
    #     # Check collisions for the food
    #     food_collision = self._goal.collidelist(self._obstacles)
    #     if food_collision != -1:
    #         raise CollisionError(f"The food {self._goal} collides with the obstacle {self._obstacles[food_collision]}.")

    ## Properties
    @property
    def snake(self):
        "Snake agent represented by a subclass of Snake."
        if self._snake is None:
            raise InitialisationError("The snake is not initialised. Reset the environment first !")
        return self._snake

    @property
    def observations(self):
        "Observation associated with the current state of the environment"
        if self._position is None:
            raise InitialisationError("The position is not initialised. Reset the environment first !")
        left, front, right = self._get_neighbours()

        return np.array([
            ## Neighbours collision
            self._is_collision(left), # LEFT
            self._is_collision(front), # FRONT
            self._is_collision(right), # RIGHT
            ## Snake direction
            self._snake.direction == Direction.NORTH, # UP
            self._snake.direction == Direction.EAST, # RIGHT
            self._snake.direction == Direction.SOUTH, # DOWN
            self._snake.direction == Direction.WEST, # LEFT
            ## Food position
            self._goal.y < self._position.y, # UP
            self._goal.x > self._position.x, # RIGHT
            self._goal.y > self._position.y, # DOWN
            self._goal.x < self._position.x, # LEFT
        ], dtype=int)

    #TODO : Think about a new way to initialize snake !
    def _place_snake(self) -> Snake:
        # As the snake is initialised along
        # available_positions = [(x, y) for x in range(2, self.width) for y in range(self.height) if all(self._free_positions[x-2:x+1, y])]
        # assert len(available_positions) > 0, "There is no available positions for the snake in the current environment"
        available_positions = self.free_positions
        self._rng.shuffle(available_positions)
        snake_positions = self._get_snake_positions(available_positions)
        # x, y = self.rng.choice(self.free_positions)
        # snake = SnakeAI(x * self.pixel, y * self.pixel, pixel=self.pixel)
        snake = Snake(snake_positions, pixel=self.pixel)
        # self._free_position_mask[x-2:x+1, y-2:y+1] = False
        return snake

    def _get_snake_positions(self, available_positions : List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        for x, y in available_positions:
            # RIGHT
            if (x +1 , y) in available_positions:
                if (x + 1, y - 1) in available_positions:
                    return [(x, y), (x + 1, y), (x + 1, y - 1)]
                if (x + 2 , y) in available_positions:
                    return [(x, y), (x + 1, y), (x + 2, y)]
                if (x + 1, y + 1) in available_positions:
                    return [(x, y), (x + 1, y), (x + 1, y + 1)]
            # BOTTOM
            if (x , y + 1) in available_positions:
                if (x + 1 , y + 1) in available_positions:
                    return [(x, y), (x, y + 1), (x + 1, y + 1)]
                if (x, y + 2) in available_positions:
                    return [(x, y), (x, y + 1), (x, y + 2)]
                if (x - 1, y + 1) in available_positions:
                    return [(x, y), (x, y + 1), (x - 1, y + 1)]
            # LEFT
            if (x - 1, y) in available_positions:
                if (x - 1 , y + 1) in available_positions:
                    return [(x, y), (x - 1, y), (x - 1, y + 1)]
                if (x - 2, y) in available_positions:
                    return [(x, y), (x - 1, y), (x - 2, y)]
                if (x - 1, y - 1) in available_positions:
                    return [(x, y), (x - 1, y), (x - 1, y - 1)]
            # TOP
            if (x , y - 1) in available_positions:
                if (x - 1 , y - 1) in available_positions:
                    return [(x, y), (x, y - 1), (x - 1, y - 1)]
                if (x, y - 2) in available_positions:
                    return [(x, y), (x, y - 1), (x, y - 2)]
                if (x + 1, y - 1) in available_positions:
                    return [(x, y), (x, y - 1), (x + 1, y - 1)]
        raise ConfigurationError("There is no valid configuration in free space to place a 3 pixel snake.")

    # Collision handling
    def _collide_with_snake_body(self, rect: Optional[pygame.Rect] = None) -> bool:
        if rect is None:
            return self._snake.collide_with_itself()
        return rect.collidelist(self._snake.body) != -1

    def _is_collision(self, rect: Optional[pygame.Rect] = None) -> bool:
        return super()._is_collision(rect) or self._collide_with_snake_body(rect)

    def _get_neighbours(self) -> List[pygame.Rect]:
        """Return left, front and right neighbouring bounding boxes located at snake head

        Raises:
            ValueError: if the snake direction is not in [UP, RIGHT, DOWN, LEFT]

        Returns:
            List[pygame.Rect]: clockwise list of neighbours positions in the snake coordinates, from left to right
        """
        up, right, down, left = super()._get_neighbours()

        if self._snake.direction == Direction.NORTH:
            return left, up, right
        if self._snake.direction == Direction.EAST:
            return up, right, down
        if self._snake.direction == Direction.SOUTH:
            return right, down, left
        if self._snake.direction == Direction.WEST:
            return down, left, up
        raise ValueError(f'Unknown direction {self._snake.direction}')

    ## Dunder methods
    def __eq__(self, other: object) -> bool:
        assert isinstance(other, SnakeEnv), f"Can not compare equality with an instance of {type(other)}. Expected type is SnakeEnv"
        grid_world_check = super().__eq__(other)
        snake_check = self.snake == other.snake
        return grid_world_check and snake_check

    def __repr__(self) -> str:
        return f"{__class__.__name__}(width={self.width!r}, height={self.height!r}, pixel={self.pixel!r}, nb_obstacles={self.nb_obs!r}, max_obs_size={self._max_obs_size!r}, render_mode={self.render_mode!r}, seed={self._seed})"

if __name__ == "__main__":
    import time
    snake_env = SnakeEnv(20, 20, nb_obs=10, max_obs_size=10, render_mode="human")
    snake_env.reset()
    fps = 10

    action = 0
    done = False
    while not done:
        # time.sleep(1/fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                snake_env.close()
                quit()
            key_pressed = event.type == pygame.KEYDOWN
            if key_pressed and event.key == pygame.K_UP:
                action = 0
            if key_pressed and event.key == pygame.K_RIGHT:
                action = 1
            # if key_pressed and event.key == pygame.K_DOWN:
            #     action = 2
            if key_pressed and event.key == pygame.K_LEFT:
                action = 2
        _, _, terminated, _ = snake_env.step(action)
        if terminated:
            snake_env.reset()
            print('You suck ! Try again !')
        snake_env.render()