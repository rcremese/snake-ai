from snake_ai.envs.grid_world import GridWorld
from snake_ai.envs.walker import Walker2D
from snake_ai.envs.geometry import Rectangle
import pygame

import numpy as np

from snake_ai.utils import errors
from typing import List, Optional, Tuple, Dict, Any


class RandomObstaclesEnv(GridWorld):
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        pixel: int = 10,
        nb_obs: int = 0,
        max_obs_size: int = 1,
        seed: int = 0,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            width (int, optional): Environment width in terms of `metapixel`. Defaults to 20.
            height (int, optional): Environment height in terms of `metapixel`. Defaults to 20.
            pixel (int, optional): Size of the environment `metapixel` in terms of classical pixel. Defaults to 10.
            nb_obs (int, optional): Number of obstacles in the environment. Defaults to 0.
            max_obs_size (int, optional): Maximum obstacle size in terms of `metapixel`. Defaults to 1.
            seed (int, optional): Seed for the Random Number Generator (RNG) of the environment (numpy.random). Defaults to 0.
            render_mode (str, optional): Name of the mode the environment should be rendered. If None, there is no rendering. Defaults to None.
        Raises:
            TypeError: Raised if the inputs are not instance of ints.
            ValueError: Raised if the inputs are negative ints or the render mode is not one of the None,
            ValueError: Raised if the render mode is not one of the following : None, "human", "rgb_array"
        """
        super().__init__(width, height, pixel, seed, render_mode)
        # Initialise all numerical parameters
        if (max_obs_size < 1) or (nb_obs < 0):
            raise ValueError(
                f"Only positive integers are allowed for parameters nb_obs & max_obs_size (>= 1) . Get ({nb_obs}, {max_obs_size})"
            )
        if (max_obs_size > self.width) or (max_obs_size > self.height):
            raise ValueError(
                f"Can not set an obstacle of size {max_obs_size} in an environment of size ({self.width}, {self.height})"
            )
        if nb_obs * max_obs_size**2 > self.width * self.height - 2:
            raise errors.ConfigurationError(
                f"The maximal area of obstacles is greater than the maximal area of the environment."
            )
        self._nb_obs, self._max_obs_size = int(nb_obs), int(max_obs_size)
        self._obs_sizes = self._rng.integers(
            1, max_obs_size, size=nb_obs, endpoint=True
        )

    ## Public methods
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Method to reset the environment

        Args:
            seed (Optional[int], optional): Seed to be used for environment generation. If None, use the previously defined seed, otherwise overwrite the seed value. Defaults to None.

        Returns:
            observations, info: Vector of 0 and 1 that represent the observations and dictionary of available informations
        """
        super().reset(seed)
        self.seed(seed)
        # Initialise score and a grid of free positions
        self.score = 0
        self.truncated = False
        self._free_position_mask = np.ones((self.width, self.height))
        # Initialise obstacles
        if self._nb_obs > 0:
            self._populate_grid_with_obstacles()
        else:
            self._obstacles = []
        # Initialise goal & position
        x_goal, y_goal = self._rng.choice(self.free_positions)
        self.goal = Rectangle(
            x_goal * self.pixel, y_goal * self.pixel, self.pixel, self.pixel
        )
        x_agent, y_agent = self._rng.choice(self.free_positions)
        self.agent = Walker2D(x_agent, y_agent, self.pixel)
        return self.observations, self.info

    ## Properties
    @GridWorld.name.getter
    def name(self) -> str:
        return f"RandomObstacles({self.width},{self.height})"

    @GridWorld.obstacles.setter
    def obstacles(self, rectangles: List[Rectangle]):
        # Simple case in which the user provide only 1 rectangle
        if isinstance(rectangles, Rectangle):
            rectangles = [rectangles]
        # Make sure all the rectangles fit with the environment
        for rect in rectangles:
            self._sanity_check(rect)
        # free all positions of current obstacles
        if self._obstacles is not None:
            for obstacle in self._obstacles:
                x, y = self._get_grid_position(obstacle)
                size = obstacle.width // self.pixel
                self._free_position_mask[x : x + size, y : y + size] = True
        # set new obstacles positions to false
        for rectangle in rectangles:
            x, y = self._get_grid_position(rectangle)
            size = rectangle.width // self.pixel
            self._free_position_mask[x : x + size, y : y + size] = False
        # TODO : Implement a warning if obstacles overlaps with goal or agent !
        self._obstacles = rectangles

    ## Private methods
    def _populate_grid_with_obstacles(self):
        self._obstacles = []
        for size in self._obs_sizes:
            obstacle = self._place_obstacle(size)
            # Update free_position_mask
            x, y = self._get_grid_position(obstacle)
            self._free_position_mask[x : x + size, y : y + size] = False
            # Append obstacle to the list
            self._obstacles.append(obstacle)

    def _place_obstacle(self, size: int) -> Rectangle:
        """Place an obstacle of a given size in the environment while repecting the free position condition

        Args:
            size (int): size of the obstacle

        Returns:
            Rectangle: Obstacle to place in the environment, represented as a square of size 1.
        """
        available_positions = self.free_positions
        self._rng.shuffle(available_positions)
        for x, y in available_positions:
            if self._free_position_mask[x : x + size, y : y + size].all():
                return Rectangle(
                    x * self.pixel, y * self.pixel, size * self.pixel, size * self.pixel
                )
        raise errors.ConfigurationError(
            f"Unable to place obstacle of size {size} in the environment. Reduce nb_obs or max_obs_size."
        )

    # TODO : cleaner le code
    # def _place_obstacle(self, size: int) -> Rectangle:
    #     """Place an obstacle of a given size in the environment while repecting the free position condition

    #     Args:
    #         size (int): size of the obstacle

    #     Returns:
    #         Rectangle: Obstacle to place in the environment, represented as a square of size 1.
    #     """
    #     assert size > 0, f"Obstacle size need to be at least 1. Get {size}"
    #     # available_positions = [(x, y) for x in range(self.width-(size-1)) for y in range(self.height-(size-1)) if self._free_positions[x, y]]
    #     # assert len(available_positions) > 0, f"There is no available position for an obstacle of size {size}"
    #     x, y = self._rng.choice(self.free_positions)
    #     obstacle = Rectangle(x * self.pixel, y * self.pixel, size * self.pixel, size * self.pixel)
    #     # Remove all possible
    #     self._free_position_mask[x:x+size, y:y+size] = False
    #     # if size > 1:
    #     #     self._free_position_mask[x:x+size, y:y+size] = False
    #     # else:
    #     #     self._free_position_mask[x, y] = False
    #     return obstacle

    ## Dunder methods
    def __repr__(self) -> str:
        return f"{__class__.__name__}(width={self.width}, height={self.height}, pixel={self.pixel}, nb_obs={self._nb_obs}, max_obs_size={self._max_obs_size}, render_mode={self.render_mode}, seed={self._seed})"


if __name__ == "__main__":
    import time

    snake_env = RandomObstaclesEnv(
        20, 20, nb_obs=15, max_obs_size=5, render_mode="human"
    )
    seed = 0
    snake_env.reset(seed)
    print(snake_env._free_position_mask)
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
