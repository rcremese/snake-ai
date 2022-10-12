from turtle import right

from matplotlib.sankey import RIGHT
from snake_ai.envs.snake_2d_env import PIXEL_SIZE, Snake2dEnv
from snake_ai.envs.line import Line
from snake_ai.envs.utils import Direction
from snake_ai.wrappers.snake_relative_position import SnakeRelativePosition
import numpy as np

class TestSnakeEnv():
    def test_observations(self):
        env = Snake2dEnv(width=5, height=5)
        wrapped_env = SnakeRelativePosition(env)
        observations, info = wrapped_env.reset()
        directions = wrapped_env._get_list_direction(info['snake_direction'])
        passing_matrix = wrapped_env._get_passing_matrix(info['snake_direction'])
        assert info['snake_direction'] == Direction.RIGHT
        assert info['collision_lines'][Direction.LEFT] is None
        assert len(observations) == 8

        dist_center_border = 2.5 * PIXEL_SIZE
        assert np.array_equal(observations[0], np.array([0, dist_center_border])), f"The observable for direction {directions[0]} is {observations[0]}" # RIGHT
        assert np.array_equal(observations[1], np.array([dist_center_border, dist_center_border])), f"The observable for direction {directions[1]} is {observations[1]}" # DOWN_RIGHT
        assert np.array_equal(observations[2], np.array([dist_center_border, 0])), f"The observable for direction {directions[2]} is {observations[2]}" # DOWN
        assert np.array_equal(observations[3], np.array([dist_center_border, -dist_center_border])), f"The observable for direction {directions[3]} is {observations[3]}" # DOWN_LEFT
        assert np.array_equal(observations[4], np.array([-dist_center_border, -dist_center_border])), f"The observable for direction {directions[4]} is {observations[4]}" # UP_LEFT
        assert np.array_equal(observations[5], np.array([-dist_center_border, 0])), f"The observable for direction {directions[4]} is {observations[5]}" # UP
        assert np.array_equal(observations[6], np.array([-dist_center_border, dist_center_border])), f"The observable for direction {directions[6]} is {observations[6]}" # UP_RIGHT
        food = Line(info['snake_head'], info['food'])
        local_food = np.matmul(passing_matrix, food.to_vector())
        assert np.array_equal(observations[7], local_food), f"The food vector for food located at {info['food']} is {observations[7]}" # FOOD

    def test_passing_matrix(self):
        env = Snake2dEnv(width=5, height=5)
        wrapped_env = SnakeRelativePosition(env)
        for direction in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
            passing_matrix = wrapped_env._get_passing_matrix(direction)
            if direction == Direction.UP:
                assert np.array_equal(passing_matrix, np.array([[1, 0], [0, -1]]))
            if direction == Direction.RIGHT:
                assert np.array_equal(passing_matrix, np.array([[0, 1], [1, 0]]))
            if direction == Direction.DOWN:
                assert np.array_equal(passing_matrix, np.array([[-1, 0], [0, 1]]))
            if direction == Direction.LEFT:
                assert np.array_equal(passing_matrix, np.array([[0, -1], [-1, 0]]))