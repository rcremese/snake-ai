from snake_ai.envs.grid_world_3d import GridWorld3D
from snake_ai.envs.geometry import Cube
from snake_ai.utils import errors

import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class RandomObstacles3D(GridWorld3D):
    def __init__(
        self,
        width: int,
        height: int,
        depth: int,
        nb_obs: int = 0,
        max_size: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__(width, height, depth, seed)
        if (max_size < 1) or (nb_obs < 0):
            raise ValueError(
                f"Only positive integers are allowed for parameters nb_obs & max_size (>= 1) . Get ({nb_obs}, {max_size})"
            )
        if (
            (max_size > self.width)
            or (max_size > self.height)
            or (max_size > self.depth)
        ):
            raise ValueError(
                f"Can not set an obstacle of size {max_size} in an environment of size ({self.width}, {self.height}, {self.depth})"
            )
        if nb_obs * max_size**3 > self.width * self.height * self.depth:
            raise errors.ConfigurationError(
                f"The maximal area of obstacles is greater than the maximal area of the environment."
            )
        self._nb_obs, self._max_obs_size = int(nb_obs), int(max_size)

    ## Public methods
    def reset(self, seed: int = None):
        super().reset(seed)
        if self._nb_obs > 0:
            self.obstacles = self._populate_grid_with_obstacles()
        else:
            self.obstacles = []

    ## Private methods
    def _populate_grid_with_obstacles(self) -> List[Cube]:
        ## Get all available free positions
        self._free_position_mask[self.agent.x, self.agent.y, self.agent.z] = False

        obs_sizes = self._rng.integers(
            1, self._max_obs_size, size=(self._nb_obs, 3), endpoint=True
        )
        obstacles = []

        for obs_size in obs_sizes:
            ## TODO : Find a better way to do this
            free_positions = self.free_positions
            while True:
                position = self._rng.choice(free_positions)
                obstacle = Cube(*position, *obs_size)
                # Check if the obstacle is valid
                if (
                    obstacle.is_inside(self.bounds)
                    and not obstacle.contains(self.agent)
                    and not obstacle.contains(self.goal)
                    and not any(obstacle.contains(obs) for obs in obstacles)
                ):
                    break
            # Update the mask of valid positions and add the obstacle to the list
            max_pos = position + obs_size

            self._free_position_mask[
                obstacle.x : max_pos[0],
                obstacle.y : max_pos[1],
                obstacle.z : max_pos[2],
            ] = False
            obstacles.append(obstacle)

        self._free_position_mask[self.agent.x, self.agent.y, self.agent.z] = True
        return obstacles


if __name__ == "__main__":
    env = RandomObstacles3D(10, 10, 10, nb_obs=10, max_size=3)
    env.reset(10)
    print(len(env.obstacles), env.nb_obstacles)
    env.render()
