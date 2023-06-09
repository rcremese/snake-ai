from snake_ai.envs.grid_world import GridWorld
from snake_ai.envs.geometry import Rectangle
from snake_ai.envs.walker import Walker2D
from typing import Optional, Tuple, Dict, Any
from snake_ai.utils import errors
import numpy as np

class SlotEnv(GridWorld):
    def __init__(self, width: int = 20, height: int = 20, pixel: int = 10, seed: int = 0, render_mode: Optional[str] = None, **kwargs) -> None:
        if (width < 5) or (height < 5):
            raise errors.InitialisationError(f"Can not instantiate a room escape environment with width or height lower than 5. Get ({width}, {height})")
        super().__init__(width, height, pixel, seed, render_mode, **kwargs)
        self._obs_center = None

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed)
        x_obs = self._rng.integers(2, self.width - 2)
        y_obs = self._rng.integers(2, self.height - 2)
        self._obs_center = (x_obs, y_obs)

        self._free_position_mask[x_obs, :] = False
        self._free_position_mask[x_obs, y_obs - 1] = True
        self._free_position_mask[x_obs, y_obs + 1] = True

        self._obstacles = [Rectangle(x_obs * self.pixel, y * self.pixel, self.pixel, self.pixel) for y in range(self.height) if (y != y_obs - 1) and (y != y_obs + 1)]
        x_agent = self._rng.integers(0, x_obs)
        self.agent = Walker2D(x_agent, y_obs, self.pixel)
        x_goal = self._rng.integers(x_obs + 1, self.width)
        self.goal = Rectangle(x_goal * self.pixel, y_obs * self.pixel, self.pixel, self.pixel)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        _, reward, terminated, info = super().step(action)
        if info["truncated"] :
            if self.agent.position.x < self._obs_center[0] * self.pixel:
                x_goal = self._rng.integers(self._obs_center[0] + 1, self.width)
            else:
                x_goal = self._rng.integers(0, self._obs_center[0])
            self.goal = Rectangle(x_goal * self.pixel, self._obs_center[1] * self.pixel, self.pixel, self.pixel)
        return self.observations, reward, terminated, self.info

    ## Properties
    @GridWorld.name.getter
    def name(self) -> str:
        return f"Slot({self.width}, {self.height})"

if __name__ == "__main__":
    room_escape = SlotEnv(20, 20, pixel=20, render_mode="human")
    room_escape.reset(10)
    room_escape.render()
