##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-11-02 2:10:20 pm
 # @copyright https://mit-license.org/
 #

from turtle import done
from snake_ai.envs import Snake2dEnv, SnakeClassicEnv
from snake_ai.wrappers.diffusion_wrapper import DiffusionWrapper
import pygame
import numpy as np

COEF = 1_000
W, H = 20, 20
OBS = 40
MAX_STEP = 100

def main():
    env = SnakeClassicEnv(render_mode='human', width=W, height=H, nb_obstacles=OBS)
    env = DiffusionWrapper(env, diffusion_coef=COEF)
    env.metadata['render_fps'] = 20
    obs = env.reset()
    i = 0
    while True:
        i += 1
        action = np.argmax(obs)
        obs, _, done, info = env.step(action)
        if done or i > MAX_STEP:
            i=0
            obs = env.reset()
        # once food atteigned, reset i
        if info['truncated']:
            i=0
        env.render()
        # check for safe quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()



if __name__ == '__main__':
    main()