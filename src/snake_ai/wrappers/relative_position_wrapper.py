##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-10-10 5:27:10 pm
 # @copyright https://mit-license.org/
 #
import gym
from gym.utils.play import play
import gym.spaces

import numpy as np
import logging
from typing import Dict, List, Tuple
from collections import OrderedDict

from snake_ai.envs.line import Line
from snake_ai.envs.snake_2d_env import Snake2dEnv
from snake_ai.envs.snake_goal_env import SnakeGoalEnv
from snake_ai.envs.utils import Direction


class RelativePositionWrapper(gym.Wrapper):
    def __init__(self, env: Snake2dEnv):
        super().__init__(env)
        self.env : Snake2dEnv # syntaxe to work with linters in VScode
        obs_limits = np.repeat([env.window_size], self.env._nb_obs, axis=0)
        self.action_space = gym.spaces.Discrete(3)

        if isinstance(env, SnakeGoalEnv):
            self.observation_space = gym.spaces.Dict({
                "observation" : gym.spaces.Box(low = -obs_limits, high = obs_limits, shape=(self.env._nb_obs, 2)), # point cloud in local frame                "achieved_goal" : gym.spaces.Box(low=(0,0), high=self.window_size, shape=(2,0), dtype=int), # snake head position
                "achieved_goal" : gym.spaces.Box(low=np.zeros(2), high=np.array(self.env.window_size), shape=(2,), dtype=int), # snake head position
                "desired_goal" : gym.spaces.Box(low=np.zeros(2), high=np.array(self.env.window_size), shape=(2,), dtype=int), # food position
            })
        elif isinstance(env, Snake2dEnv):
            self.observation_space = gym.spaces.Box(low = -obs_limits, high = obs_limits, shape=(self.env._nb_obs, 2))
        else:
            raise TypeError("Expected Snake2dEnv or SnakeGoalEnv environment")

    def reset(self) -> Tuple[np.array, dict]:
        obs = super().reset()
        info = self.env.info
        directions = self._get_list_direction(info['snake_direction'])
        rot_matrix = self._get_passing_matrix(info['snake_direction'])
        food_line = Line(info['snake_head'], info['food'])
        observations = self._get_observables_in_local_frame(directions, rot_matrix, info['collision_lines'], food_line)
        # add goal information to observations
        if isinstance(self.observation_space, gym.spaces.Dict):
            observations = OrderedDict({
                "observation" : observations,
                "achieved_goal" : obs["achieved_goal"],
                "desired_goal" : obs["desired_goal"]
            })
        return observations

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, info = self.env.step(action)
        assert isinstance(info, dict) and set(["collision_lines", "snake_direction", "snake_head", "food"]).issubset(info.keys()), info.keys()
        directions = self._get_list_direction(info['snake_direction'])
        rot_matrix = self._get_passing_matrix(info['snake_direction'])
        food_line = Line(info['snake_head'], info['food'])
        observations = self._get_observables_in_local_frame(directions, rot_matrix, info['collision_lines'], food_line)
        # add goal information to observations
        if isinstance(self.observation_space, gym.spaces.Dict):
            observations = OrderedDict({
                "observation" : observations,
                "achieved_goal" : obs["achieved_goal"],
                "desired_goal" : obs["desired_goal"]
            })

        return observations, reward, terminated, info

    def _get_observables_in_local_frame(self, direction_list : List[Direction], rot_matrix : np.ndarray,
    collision_lines_dict : Dict[Direction, Line], food_line : Line) -> np.array:
        """Return point cloud in the agent local frame.

        The point cloud is aligned the snake moving direction and follow a clockwise patern (cf. _get_list_direction).
        Opposite direction to the moving direction is removed and food position is local frame is added to the end

        Args:
            direction_list (List[Direction]): list of all direction to consider, returned by _get_list_direction
            rot_matrix (np.ndarray): rotation matrix to convert vectors representing cloud point into the local frame
            collision_lines_dict (Dict[Direction, Line]): dict of collision lines expressed in the game frame
            food_line (Line): line between the snake head and the food in the game frame

        Returns:
            np.array: observables in the local frame
        """
        assert rot_matrix.shape == (2, 2)

        observations = np.zeros((self.env._nb_obs, 2))
        for i, direction in enumerate(direction_list):
            #TODO : check this strange behaviour
            if collision_lines_dict[direction] is None:
                observations[i,:] = np.zeros(2)
            else:
                observations[i, :] = np.matmul(rot_matrix, collision_lines_dict[direction].to_vector())
        # get the food in the snake local coordinate
        observations[-1, :] = np.matmul(rot_matrix, food_line.to_vector())
        return observations

    def _get_list_direction(self, snake_dir : Direction) -> List[Direction]:
        """Return the list of clockwise directions from the snake direction.

        The opposite direction is removed from the list. The list starts from the moving direction of the agent
        i.e : suppose the agent's direction is RIGHT, then LEFT is removed from the list and elements are added clockwise
        RIGHT -> DOWN_RIGHT -> DOWN -> DOWN_LEFT -> XXX -> UP_LEFT -> UP -> UP_RIGHT
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
    wrapped_env = RelativePositionWrapper(env)
    logging.basicConfig(level=logging.INFO)
    game = play(wrapped_env, keys_to_action={"q" : 0, "z" : 1, "d" : 2}, noop=1)
