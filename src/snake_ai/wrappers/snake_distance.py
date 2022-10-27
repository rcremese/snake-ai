import gym
from gym.spaces import Box
from snake_ai.envs import Snake2dEnv
import numpy as np

class SnakeDistance(gym.ObservationWrapper):
    def __init__(self, env: Snake2dEnv):
        super().__init__(env)
        env_shape = env.observation_space.shape
        self.observation_space = Box(low = 0, high = env.max_dist, shape=(env_shape[0], ))
        self.norm = max(env.window_size)

    def observation(self, observation):
        return np.linalg.norm(observation, axis=1) / self.norm
