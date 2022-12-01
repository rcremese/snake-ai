import argparse
import pygame

from snake_ai.utils import Line
from snake_ai.envs import SnakeClassicEnv, SnakeHuman
from snake_ai.utils import Direction

import gym
import numpy as np
from typing import Tuple, List


class PlayableSnake(gym.Wrapper):
    def __init__(self, env: SnakeClassicEnv, fps : int = 10):
        assert isinstance(env, SnakeClassicEnv)
        super().__init__(env)
        self.env.metadata['render_fps'] = fps
        self.observation_space = gym.spaces.Box(low=0, high=env.max_dist, shape=(1,))
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, **kwargs) -> Tuple[float, dict]:
        super().reset(**kwargs)
        snake_head = self.env.snake.head
        self.env.snake = SnakeHuman(*snake_head.topleft, self.env._pixel_size)

        info = self.env.info
        obs =  np.linalg.norm(Line(info['snake_head'], info['food']).to_vector())
        return obs

    def step(self, action: int) -> Tuple[np.array, float, bool, bool, dict]:
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        _, reward, terminated, info = super().step(action)

        info = self.env.info
        obs =  np.linalg.norm(Line(info['snake_head'], info['food']).to_vector())
        return obs, reward, terminated, info

    @staticmethod
    def action_from_direction(direction : Direction) -> int:
        if direction == Direction.LEFT:
            return 0
        if direction == Direction.RIGHT:
            return 1
        if direction == Direction.UP:
            return 2
        if direction == Direction.DOWN:
            return 3
        raise ValueError(f"Unknown direction {direction}. Expected LEFT, RIGHT, UP or LEFT")

    @staticmethod
    def direction_from_action(action : int) -> Direction:
        if action == 0:
            return Direction.LEFT
        if action == 1:
            return Direction.RIGHT
        if action == 2:
            return Direction.UP
        if action == 3:
            return Direction.DOWN
        raise ValueError(f"Unknown action {action}. Expected 0, 1, 2 or 3")

def play(width, height, speed, obstacles):
    env = SnakeClassicEnv(render_mode='human', width=width, height=height, nb_obstacles=obstacles)
    wrapped_env = PlayableSnake(env, fps=speed)

    pygame.init()
    wrapped_env.reset()
    info = wrapped_env.info
    while True:
        action = wrapped_env.action_from_direction(info['snake_direction'])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            key_pressed = event.type == pygame.KEYDOWN
            if key_pressed and event.key == pygame.K_LEFT:
                action = 0
            if key_pressed and event.key == pygame.K_RIGHT:
                action = 1
            if key_pressed and event.key == pygame.K_UP:
                action = 2
            if key_pressed and event.key == pygame.K_DOWN:
                action = 3
        _, _, terminated, info = wrapped_env.step(action)

        if terminated :
            # TODO : make possibility for user to replay
            print("You loose ! \nHit enter to retry or escape to quit.\n")
            # pygame.quit()
            # quit()
            wrapped_env.reset()
        wrapped_env.render('human')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--width', default=20, type=int, help='Width of the canvas in pixels')
    parser.add_argument('-d', '--height', default=20, type=int, help='Height of the canvas in pixels')
    parser.add_argument('-s', '--speed', default=10, type=int, help='Speen in FPS (the higher, the faster)')
    parser.add_argument('-o', '--obstacles', default=20, type=int, help='Number of obstacles')
    args = parser.parse_args()
    play(**vars(args))

if __name__ == '__main__':
    main()