##
# @author Robin CREMESE <robin.cremese@gmail.com>
# @file Description
# @desc Created on 2022-11-29 2:22:46 pm
# @copyright https://mit-license.org/
#
from time import sleep

import pygame
from snake_ai.wrappers import DistanceWrapper, DeterministicDiffusionWrapper
from snake_ai.wrappers.relative_position_wrapper import RelativePositionWrapper
from snake_ai.envs import SnakeClassicEnv
from stable_baselines3 import DQN
import gym

COEF = 10
SEED = 0


class LearningAgent:
    def __init__(
        self,
        env_name: str,
        model_name: str,
    ) -> None:
        pass

    def train(self):
        pass

    def test(self):
        pass


if __name__ == "__main__":
    from stable_baselines3.dqn.dqn import DQN

    train = False
    if train:
        rmode = None
    else:
        rmode = "human"
    fps = 50
    env = SnakeClassicEnv(render_mode=rmode, width=20, height=20, nb_obstacles=20)
    # env = DeterministicDiffusionWrapper(env, diffusion_coef=COEF, seed=SEED)

    if train:
        model = DQN(
            "MlpPolicy", env, verbose=1, tensorboard_log="./logs/diffusion_snake_dqn"
        )
        model.learn(total_timesteps=100_000)
        model.save("dqn_snake_env")
    else:
        model = DQN.load("dqn_snake_env")
        obs = env.reset()
        i = 0
        while True:
            i += 1
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render(rmode)
            sleep(1 / fps)
            if i > 1000 or dones:
                obs = env.reset()
                i = 0
            # Check quit action
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
