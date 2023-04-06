from snake_ai.envs.grid_world import GridWorld
from snake_ai.envs.walker import Walker2D
from snake_ai.envs.geometry import Rectangle

from mazelib import Maze
from mazelib.generate.BacktrackingGenerator import  BacktrackingGenerator
from mazelib.generate.HuntAndKill import HuntAndKill
from mazelib.generate.Kruskal import Kruskal
from mazelib.generate.Prims import Prims
from mazelib.generate.Wilsons import Wilsons
from typing import Optional, Tuple, Dict, Any
import numpy as np
import warnings

import pygame

class MazeGrid(GridWorld):
    maze_generator = ["prims", "hunt_and_kill", "backtracking", "wilsons", "kruskal"]

    def __init__(self, width: int = 20, height: int = 20, pixel: int = 10, seed: int = 0, render_mode: Optional[str] = None, maze_generator: str = "prims"):
        if maze_generator.lower() not in self.maze_generator:
            raise ValueError(f"Unknown maze generator {maze_generator}. Expected one of the following values : {self.maze_generator}")
        self.maze_generator = maze_generator.lower()

        if width < 5 or height < 5:
            raise ValueError(f"Can not instantiate a maze grid environment with width or height lower than 5. Get ({width}, {height})")
        if width % 2 == 0 or height % 2 == 0:
            # Get the lower odd number for width and height if they are even
            width -= 1 if (width % 2 == 0) else 0
            height -= 1 if (height % 2 == 0) else 0
            warnings.warn(f"Maze generation works with odd number. Width and height will be converted to the lower odd numbers {width} & {height}.",
                          RuntimeWarning)
        super().__init__(width, height, pixel, seed, render_mode)
        # Set the number of rows and columns for the maze generator to correspond with width and height of the grid
        rows, cols = (width + 1) // 2, (height + 1) // 2
        self._maze = Maze(seed)
        if self.maze_generator == "hunt_and_kill":
            self._maze.generator = HuntAndKill(rows, cols)
        elif self.maze_generator == "prims":
            self._maze.generator = Prims(rows, cols)
        elif self.maze_generator == "backtracking":
            self._maze.generator = BacktrackingGenerator(rows, cols)
        elif self.maze_generator == "wilsons":
            self._maze.generator = Wilsons(rows, cols)
        elif self.maze_generator == "kruskal":
            self._maze.generator = Kruskal(rows, cols)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed)
        self._maze.generate()
        obstacle_mask = np.array(self._maze.grid[1:-1, 1:-1], dtype=bool)
        self._free_position_mask = ~obstacle_mask
        # Place obstacles
        obstacle_positions = np.argwhere(obstacle_mask)
        self._obstacles = []
        for x, y in obstacle_positions:
            self._obstacles.append(Rectangle(x * self.pixel, y * self.pixel, self.pixel, self.pixel))
        # Place food
        corners = [(0, 0), (self.width - 1, 0), (self.width - 1, self.height - 1), (0, self.height - 1)]
        goal_index = self._rng.choice(4)
        self.goal = Rectangle(corners[goal_index][0] * self.pixel, corners[goal_index][1] * self.pixel, self.pixel, self.pixel)
        # Place agent
        agent_index = (goal_index + 2) % 4
        self.agent = Walker2D(*corners[agent_index], self.pixel)
        return self.observations, self.info

    def seed(self, seed: Optional[int] = None):
        Maze.set_seed(seed)
        super().seed(seed)

    ## Dunder methods
    def __repr__(self):
        return f"{__class__.__name__}(width={self.width}, height={self.height}, pixel={self.pixel}, seed={self._seed}, render_mode={self.render_mode}, maze_generator={self.maze_generator})"

if __name__ == "__main__":
    maze = MazeGrid(25, 21, 20, 0, "human", "backtracking")
    maze.reset()
    maze.render()

    seed = 0
    done = False
    while not done:
        # time.sleep(1/fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                maze.close()
                quit()
            key_pressed = event.type == pygame.KEYDOWN
            if key_pressed and event.key == pygame.K_RIGHT:
                seed += 1
                maze.reset(seed)
                maze.render()
