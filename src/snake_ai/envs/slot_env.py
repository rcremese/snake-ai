from snake_ai.envs.grid_world import GridWorld
from snake_ai.envs.geometry import Rectangle
from snake_ai.envs.walker import Walker2D
from typing import Optional, Tuple, Dict, Any
from snake_ai.utils import errors
import numpy as np


class SlotEnv(GridWorld):
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        pixel: int = 10,
        seed: int = 0,
        render_mode: Optional[str] = None,
        **kwargs,
    ) -> None:
        if (width < 5) or (height < 5):
            raise errors.InitialisationError(
                f"Can not instantiate a room escape environment with width or height lower than 5. Get ({width}, {height})"
            )
        super().__init__(width, height, pixel, seed, render_mode, **kwargs)
        self._obs_center = None
        self._entries = None

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed)
        x_obs = self._rng.integers(2, self.width - 2)
        y_obs = self._rng.integers(2, self.height - 2)
        self._obs_center = (x_obs, y_obs)

        self._free_position_mask[x_obs, :] = False
        self._free_position_mask[x_obs, y_obs - 1] = True
        self._free_position_mask[x_obs, y_obs + 1] = True

        self._entries = [(x_obs, y_obs - 1), (x_obs, y_obs + 1)]

        self._obstacles = [
            Rectangle(x_obs * self.pixel, y * self.pixel, self.pixel, self.pixel)
            for y in range(self.height)
            if (y != y_obs - 1) and (y != y_obs + 1)
        ]
        x_agent = self._rng.integers(0, x_obs)
        self.agent = Walker2D(x_agent, y_obs, self.pixel)
        x_goal = self._rng.integers(x_obs + 1, self.width)
        self.goal = Rectangle(
            x_goal * self.pixel, y_obs * self.pixel, self.pixel, self.pixel
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        _, reward, terminated, info = super().step(action)
        if info["truncated"]:
            if self.agent.position.x < self._obs_center[0] * self.pixel:
                x_goal = self._rng.integers(self._obs_center[0] + 1, self.width)
            else:
                x_goal = self._rng.integers(0, self._obs_center[0])
            self.goal = Rectangle(
                x_goal * self.pixel,
                self._obs_center[1] * self.pixel,
                self.pixel,
                self.pixel,
            )
        return self.observations, reward, terminated, self.info

    def close_entry(self) -> None:
        assert self._entries is not None, "The environment has not been reset"
        if len(self._entries) == 0:
            raise errors.InitialisationError(
                "All entries have been closed. Reset the environment to open new entries"
            )
        idx = np.random.randint(0, len(self._entries))

        closed_entry = self._entries.pop(idx)
        self._free_position_mask[closed_entry] = False
        self._obstacles.append(
            Rectangle(
                closed_entry[0] * self.pixel,
                closed_entry[1] * self.pixel,
                self.pixel,
                self.pixel,
            )
        )

    ## Properties
    @GridWorld.name.getter
    def name(self) -> str:
        return f"Slot({self.width},{self.height})"


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    room_escape = SlotEnv(20, 20, pixel=20, render_mode="human")
    room_escape.reset(10)
    room_escape.close_entry()
    room_escape.render()
    plt.imshow(room_escape._free_position_mask)
    plt.show()
