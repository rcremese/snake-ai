from typing import Dict
from snake_ai.envs import Snake2dEnv
from snake_ai.wrappers.distance_wrapper import DistanceWrapper
from stable_baselines3.common.env_checker import check_env
from snake_ai.wrappers.relative_position_wrapper import RelativePositionWrapper

env = Snake2dEnv(render_mode='human')
wrapped_env = RelativePositionWrapper(env)
wrapped_env_bis = DistanceWrapper(env)
# check_env(env)
# check_env(wrapped_env)
check_env(wrapped_env_bis)