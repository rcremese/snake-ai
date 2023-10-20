##
# @author Robin CREMESE <robin.cremese@gmail.com>
# @file Description
# @desc Created on 2022-11-02 2:10:20 pm
# @copyright https://mit-license.org/
#

from snake_ai.envs.snake_classic_env import SnakeClassicEnv
from snake_ai.wrappers import StochasticDiffusionWrapper
import pygame
import numpy as np

COEF = 100
W, H = 20, 20
OBS = 10
MAX_STEP = 100
NB_PART = 1e5
T_MAX = 100
SEED = 0


def main():
    env = SnakeClassicEnv(render_mode="human", width=W, height=H, nb_obstacles=OBS)
    # env = DeterministicDiffusionWrapper(env, diffusion_coef=COEF, seed=SEED)
    env = StochasticDiffusionWrapper(
        env, diffusion_coef=COEF, nb_part=NB_PART, t_max=T_MAX, seed=SEED
    )
    env.metadata["render_fps"] = 20
    obs = env.reset()
    i = 0
    while True:
        i += 1
        action = np.argmax(obs)
        obs, _, done, info = env.step(action)
        if done or i > MAX_STEP:
            i = 0
            obs = env.reset()
        # once food atteigned, reset i
        if info["truncated"]:
            i = 0
        env.render()
        # check for safe quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()


if __name__ == "__main__":
    main()
