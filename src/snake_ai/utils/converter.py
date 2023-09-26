from snake_ai.envs import GridWorld, Rectangle
import numpy as np

from typing import Union, Tuple, List


def convert_free_space_to_point_cloud(env: GridWorld) -> np.ndarray:
    """Transform free space of the environment into a point cloud.

    Args:
        env (GridWorld): Environment in 2D containing free positons.
    Returns:
        np.ndarray: _description_
    """
    assert isinstance(env, GridWorld), "environmnent must be of type GridWorld"
    assert env.free_positions is not None, "Environment does not contain free positions"
    return np.array(env.free_positions) + 0.5


def convert_obstacles_to_binary_map(env: GridWorld, res: str = "pixel") -> np.ndarray:
    assert isinstance(env, GridWorld), "environmnent must be of type GridWorld"
    assert env.obstacles is not None, "Environment does not contain obstacles"
    assert res.lower() in [
        "pixel",
        "meta",
    ], "Resolution must be either 'pixel' or 'meta'"
    resolution = 1 if res.lower() == "pixel" else env.pixel
    binary_map = np.zeros((env.height * resolution, env.width * resolution))
    for obstacle in env.obstacles:
        x = int(obstacle.x / env.pixel) * resolution
        y = int(obstacle.y / env.pixel) * resolution
        width = int(obstacle.width / env.pixel) * resolution
        height = int(obstacle.height / env.pixel) * resolution
        binary_map[y : y + height, x : x + width] = 1
    return binary_map


def convert_agent_position(env: GridWorld) -> np.ndarray:
    assert isinstance(env, GridWorld), "environmnent must be of type GridWorld"
    assert (
        env.agent is not None
    ), "The agent is not initialized in the environment. Call reset first."
    return np.array(env.agent.position.center) / np.array([env.pixel, env.pixel])


def convert_goal_position(env: GridWorld) -> np.ndarray:
    assert isinstance(env, GridWorld), "environmnent must be of type GridWorld"
    assert (
        env.goal is not None
    ), "The goal is not initialized in the environment. Call reset first."
    return np.array(env.goal.center) / np.array([env.pixel, env.pixel])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from snake_ai.envs import RandomObstaclesEnv

    env = RandomObstaclesEnv(width=20, height=10, pixel=10, nb_obs=10, max_obs_size=3)
    env.reset()

    point_cloud = convert_free_space_to_point_cloud(env)
    obstacles = convert_obstacles_to_binary_map(env)
    agent_position = convert_agent_position(env)
    goal_position = convert_goal_position(env)

    plt.imshow(obstacles, extent=[0, env.width, env.height, 0])
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], c="r")
    plt.scatter(agent_position[0], agent_position[1], c="b")
    plt.scatter(goal_position[0], goal_position[1], c="g")
    plt.show()
