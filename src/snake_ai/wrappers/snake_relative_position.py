##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-10-10 5:27:10 pm
 # @copyright https://mit-license.org/
 #
import gym
from gym.utils.play import play
import numpy as np
import logging
from typing import Dict, List, Tuple

from snake_ai.envs.line import Line
from snake_ai.envs.snake_2d_env import Snake2dEnv
from snake_ai.envs.utils import Direction
NB_OBS = 8

class SnakeRelativePosition(gym.Wrapper):
    def __init__(self, env: Snake2dEnv):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low = np.zeros((NB_OBS, 2)), high = np.repeat([env.window_size], NB_OBS, axis=0), shape=(NB_OBS, 2), dtype=float)
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, **kwargs) -> Tuple[np.array, dict]:
        _, info = super().reset(**kwargs)
        assert isinstance(info, dict) and set(["collision_lines", "snake_direction", "snake_head", "food"]).issubset(info.keys()), info.keys()
        directions = self._get_list_direction(info['snake_direction'])
        rot_matrix = self._get_passing_matrix(info['snake_direction'])
        food = Line(info['snake_head'], info['food'])
        observations = self._get_observables_in_local_frame(directions, rot_matrix, info['collision_lines'], food)
        return observations, info

    def step(self, action: int) -> Tuple[np.array, float, bool, bool, dict]:
        _, reward, terminated, truncated, info = self.env.step(action)
        assert isinstance(info, dict) and set(["collision_lines", "snake_direction", "snake_head", "food"]).issubset(info.keys()), info.keys()
        directions = self._get_list_direction(info['snake_direction'])
        rot_matrix = self._get_passing_matrix(info['snake_direction'])
        food = Line(info['snake_head'], info['food'])
        observations = self._get_observables_in_local_frame(directions, rot_matrix, info['collision_lines'], food)

        return observations, reward, terminated, truncated, info

    def _get_observables_in_local_frame(self, direction_list : List[Direction], rot_matrix : np.array, collision_lines_dict : Dict[Direction, Line], food : Line) -> np.array:
        assert rot_matrix.shape == (2, 2)

        observations = np.zeros((NB_OBS, 2))
        for i, direction in enumerate(direction_list):
            observations[i, :] = np.matmul(rot_matrix, collision_lines_dict[direction].to_vector())
        # get the food in the snake local coordinate
        observations[-1, :] = np.matmul(rot_matrix, food.to_vector())
        return observations

    def _get_list_direction(self, snake_dir : Direction) -> List[Direction]:
        """return the list of clockwise directions from the snake direction

        The opposite direction is removed from the list
        """
        if snake_dir == Direction.UP:
            return [Direction.UP, Direction.UP_RIGHT, Direction.RIGHT, Direction.DOWN_RIGHT, Direction.DOWN_LEFT, Direction.LEFT, Direction.UP_LEFT]
        if snake_dir == Direction.RIGHT:
            return [Direction.RIGHT, Direction.DOWN_RIGHT, Direction.DOWN, Direction.DOWN_LEFT, Direction.UP_LEFT, Direction.UP, Direction.UP_RIGHT]
        if snake_dir == Direction.DOWN:
            return [Direction.DOWN, Direction.DOWN_LEFT, Direction.LEFT, Direction.UP_LEFT, Direction.UP_RIGHT, Direction.RIGHT, Direction.DOWN_RIGHT]
        if snake_dir == Direction.LEFT:
            return [Direction.LEFT, Direction.UP_LEFT, Direction.UP, Direction.UP_RIGHT, Direction.DOWN_RIGHT, Direction.DOWN, Direction.DOWN_LEFT]
        raise ValueError(f'Unknown direction {snake_dir}')

    def _get_passing_matrix(self, snake_dir : Direction) -> np.array:
        """get the passing matrix to snake local frame given snake direction

        The snake local frame has y axis oriented along the direction and x axis orthogonal to the right
        Direction = UP :
        Snake frame        canvas  __ x frame
        y |__ x                  y|

        Args:
            snake_dir (Direction): _description_

        Returns:
            np.array: _description_
        """
        if snake_dir == Direction.UP:
            return np.array([[1, 0], [0, -1]])
        if snake_dir == Direction.RIGHT:
            return np.array([[0, 1], [1, 0]])
        if snake_dir == Direction.DOWN:
            return np.array([[-1, 0], [0, 1]])
        if snake_dir == Direction.LEFT:
            return np.array([[0, -1], [-1, 0]])
        raise ValueError(f'Unknown direction {snake_dir}')

if __name__ == '__main__':
    env = gym.make('Snake-v0', render_mode='rgb_array', nb_obstacles = 10)
    wrapped_env = SnakeRelativePosition(env)
    logging.basicConfig(level=logging.INFO)
    game = play(wrapped_env, keys_to_action={"q" : 0, "z" : 1, "d" : 2}, noop=1)
