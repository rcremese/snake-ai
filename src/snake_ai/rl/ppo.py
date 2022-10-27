from typing import Dict
from snake_ai.envs import Snake2dEnv
from snake_ai.wrappers.snake_distance import SnakeDistance
from stable_baselines3.common.env_checker import check_env
from snake_ai.wrappers.snake_relative_position import SnakeRelativePosition

env = Snake2dEnv(render_mode='human')
wrapped_env = SnakeRelativePosition(env)
wrapped_env_bis = SnakeDistance(env)
# check_env(env)
# check_env(wrapped_env)
check_env(wrapped_env_bis)