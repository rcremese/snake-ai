from snake_ai.envs.geometry import Cube
from snake_ai.utils import errors

import gymnasium as gym
import numpy as np

from typing import Any, List, Tuple
import taichi as ti


class GridWorld3D(gym.Env):
    def __init__(self, width: int, height: int, depth: int, seed: int = 0) -> None:
        assert (
            width > 0 and height > 0 and depth > 0
        ), "Width, height and depth must be positive integers"
        self.width = width
        self.height = height
        self.depth = depth
        self.bounds = Cube(0, 0, 0, width, height, depth)

        self.seed = seed
        self._free_position_mask = np.ones(
            (self.width, self.height, self.depth), dtype=bool
        )
        self.agent = None
        self.goal = None
        self.obstacles = None

    def reset(self, seed: int = None):
        if seed is not None:
            self.seed = seed

        goal_position = self._rng.integers(
            0, high=[self.width, self.height, self.width], size=3
        )
        self.goal = Cube(*goal_position, 1, 1, 1)
        self._free_position_mask[tuple(goal_position)] = False

        agent_position = self._rng.choice(self.free_positions)
        self.agent = Cube(*agent_position, 1, 1, 1)
        self.obstacles = []

    def render(self, window_size: Tuple[int] = (1280, 720)):
        assert (
            self.agent is not None
        ), "The agent is not initialised. Reset the environment first !"

        ti.init()
        goal_vert, goal_ind = convert_cube_to_wireframe(self.agent)
        agent_vert, agent_ind = convert_cube_to_wireframe(self.goal)
        bound_vert, bound_ind = convert_cube_to_wireframe(self.bounds)
        if self.nb_obstacles > 0:
            obs_vert, obs_ind = convert_cubes_to_wireframe(self.obstacles)

        center = self.center

        window = ti.ui.Window("Environment representation", window_size, fps_limit=60)
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()
        camera.position(center[0], center[1], -self.depth)
        camera.lookat(*center)

        while window.running:
            camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
            scene.set_camera(camera)
            scene.ambient_light((0.8, 0.8, 0.8))
            scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

            # Draw 3d-lines in the scene
            scene.lines(goal_vert, indices=goal_ind, color=(0.0, 1.0, 0.0), width=1.0)
            scene.lines(bound_vert, indices=bound_ind, color=(0.5, 0.5, 0.5), width=5.0)
            scene.lines(agent_vert, indices=agent_ind, color=(0.0, 0.0, 1.0), width=1.0)
            if self.nb_obstacles > 0:
                scene.lines(obs_vert, indices=obs_ind, color=(1.0, 0.0, 0.0), width=1.0)

            canvas.scene(scene)
            window.show()

    ## Properties
    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed: int):
        assert isinstance(seed, int), "Seed must be an integer"
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def free_positions(self) -> List[Tuple[int, int]]:
        """Available free positions represented as a list of position tuple (x, y) taken from the free position mask"""
        if self._free_position_mask is None:
            raise errors.InitialisationError(
                "The free position mask is not initialised. Reset the environment first !"
            )
        return np.argwhere(self._free_position_mask).tolist()

    @property
    def center(self) -> np.ndarray:
        """Center of the environment"""
        return self.bounds.center

    @property
    def nb_obstacles(self):
        """Number of obstacles in the environment"""
        return len(self.obstacles)


def convert_cube_to_wireframe(cube: Cube):
    assert isinstance(cube, Cube), "cube must be an instance of Cube"

    vertice = ti.Vector.field(3, dtype=ti.f32, shape=8)
    index = ti.Vector.field(2, dtype=ti.i32, shape=12)
    vertice.from_numpy(cube.vertices)
    index.from_numpy(cube.edges)
    return vertice, index


def convert_cubes_to_wireframe(cubes: List[Cube]):
    assert isinstance(cubes, List) and all(
        isinstance(cube, Cube) for cube in cubes
    ), f"cubes must be a list of Cube. Get {type(cubes)} of {[type(cube) for cube in cubes]}"
    nb_cubes = len(cubes)

    vertices = ti.Vector.field(3, dtype=ti.f32, shape=8 * nb_cubes)
    indexes = ti.Vector.field(2, dtype=ti.i32, shape=12 * nb_cubes)

    vert_array = np.zeros((8 * nb_cubes, 3), dtype=float)
    ind_array = np.zeros((12 * nb_cubes, 2), dtype=int)
    for i, cube in enumerate(cubes):
        vert_array[8 * i : 8 * (i + 1)] = cube.vertices
        ind_array[12 * i : 12 * (i + 1)] = 8 * i + cube.edges
    vertices.from_numpy(vert_array)
    indexes.from_numpy(ind_array)

    return vertices, indexes


if __name__ == "__main__":
    gridworld = GridWorld3D(10, 10, 10)
    gridworld.reset(10)
    gridworld.render()
