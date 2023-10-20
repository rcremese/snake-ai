##
# @author  <robin.cremese@gmail.com>
# @file Implmentation of the room escape environment
# @desc Created on 2023-04-04 11:03:10 am
# @copyright MIT License
#
from snake_ai.envs.grid_world import GridWorld
from snake_ai.envs.geometry import Rectangle
from snake_ai.envs.walker import Walker2D
from typing import Optional, Tuple, Dict, Any, List
from snake_ai.utils import errors
import numpy as np
import pygame


class RoomEscape(GridWorld):
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        pixel: int = 10,
        seed: int = 0,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        if (width < 5) or (height < 5):
            raise errors.InitialisationError(
                f"Can not instantiate a room escape environment with width or height lower than 5. Get ({width}, {height})"
            )
        super().__init__(width, height, pixel, seed, render_mode)
        # All non-instanciated attributes
        self._rooms = None

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed)
        self._populate_grid_with_obstacles()
        x_agent, y_agent = self._rng.choice(self.free_positions)
        self.agent = Walker2D(x_agent, y_agent, self.pixel)
        # self._place_goal()
        x_goal, y_goal = self._rng.choice(self.free_positions)
        self.goal = Rectangle(
            x_goal * self.pixel, y_goal * self.pixel, self.pixel, self.pixel
        )

        return self.observations, self.info

    ## Properties
    @GridWorld.name.getter
    def name(self) -> str:
        return f"RoomEscape_{self.width}x{self.height}"

    ## Private methods
    def _populate_grid_with_obstacles(self):
        # Get walls x and y coordinates (in grid units)
        x_0 = self._rng.integers(2, self.width - 2)
        self._free_position_mask[x_0, :] = False
        y_0 = self._rng.integers(2, self.height - 2)
        self._free_position_mask[:, y_0] = False
        # Store grid partition
        self._rooms = [
            Rectangle(0, 0, x_0 * self.pixel, y_0 * self.pixel),  # Top left
            Rectangle(
                (x_0 + 1) * self.pixel,
                0,
                (self.width - x_0 - 1) * self.pixel,
                y_0 * self.pixel,
            ),  # Top right
            Rectangle(
                (x_0 + 1) * self.pixel,
                (y_0 + 1) * self.pixel,
                (self.width - x_0 - 1) * self.pixel,
                (self.height - y_0 - 1) * self.pixel,
            ),  # Bottom right
            Rectangle(
                0,
                (y_0 + 1) * self.pixel,
                x_0 * self.pixel,
                (self.height - y_0 - 1) * self.pixel,
            ),  # Bottom left
        ]
        # Construct holes in the walls
        hole_x_1 = self._rng.integers(1, x_0)
        hole_x_2 = self._rng.integers(x_0 + 1, self.width - 1)
        hole_y_1 = self._rng.integers(1, y_0)
        hole_y_2 = self._rng.integers(y_0 + 1, self.height - 1)
        # Update free position mask
        self._free_position_mask[hole_x_1, y_0] = True
        self._free_position_mask[hole_x_2, y_0] = True
        self._free_position_mask[x_0, hole_y_1] = True
        self._free_position_mask[x_0, hole_y_2] = True

        self._obstacles = []
        # Construct horizontal walls
        for x in range(self.width):
            if x == hole_x_1 or x == hole_x_2:
                continue
            self._obstacles.append(
                Rectangle(x * self.pixel, y_0 * self.pixel, self.pixel, self.pixel)
            )
        # Construct vertical walls
        for y in range(self.height):
            if y == hole_y_1 or y == hole_y_2:
                continue
            self._obstacles.append(
                Rectangle(x_0 * self.pixel, y * self.pixel, self.pixel, self.pixel)
            )


if __name__ == "__main__":
    room_escape = RoomEscape(20, 20, pixel=20, render_mode="human")
    seed = 0
    room_escape.reset(seed)
    room_escape.metadata["fps"] = 5

    action = 0
    done = False
    while not done:
        # time.sleep(1/fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                room_escape.close()
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
        _, _, terminated, _ = room_escape.step(action)
        if terminated:
            seed += 1
            room_escape.reset(seed)
            print("You suck ! Try again !")
        room_escape.render()
