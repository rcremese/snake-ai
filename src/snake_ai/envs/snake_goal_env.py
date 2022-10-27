from snake_ai.envs import Snake2dEnv
from stable_baselines3 import HerReplayBuffer
import gym
import gym.spaces

class SnakeGoalEnv(Snake2dEnv, gym.GoalEnv):
    def __init__(self, render_mode=None, width: int = 20, height: int = 20, nb_obstacles: int = 0, pixel: int = 20):
        super().__init__(render_mode, width, height, nb_obstacles, pixel)
        self.observation_space = gym.spaces.Dict({
            "observation" ,
            "achieved_goal",
            "desired_goal",
        })

    def reset(self):
        # check that the environement state is all right
        gym.GoalEnv.reset(self)
        return super().reset()

    def compute_reward(self, achieved_goal, desired_goal, info):
        pass