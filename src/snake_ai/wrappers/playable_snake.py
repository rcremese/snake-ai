import argparse
import pygame
from snake_ai.envs.snake_2d_env import Snake2dEnv
from snake_ai.envs.line import Line

import gym
import numpy as np
from typing import Tuple, List

from snake_ai.envs.utils import Direction

class PlayableSnake(gym.Wrapper):
    def __init__(self, env: Snake2dEnv, fps : int = 10):
        assert isinstance(env, Snake2dEnv)
        super().__init__(env)
        self.env.metadata['render_fps'] = fps
        self.observation_space = gym.spaces.Box(low=0, high=env.max_dist)
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, **kwargs) -> Tuple[float, dict]:
        _, info = super().reset(**kwargs)
        self.env.collision_lines = {}
        obs =  np.linalg.norm(Line(info['snake_head'], info['food']).to_vector())
        return obs, {'snake_direction' : self.env.snake.direction}

    def step(self, action: int) -> Tuple[np.array, float, bool, bool, dict]:
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self.direction_from_action(action)
        self.env.snake.move(direction)
        # direction = self._action_to_direction[action]
        agent_location = self.env.snake.head.center
        target_location = self.env.food.center
        # An episode is done iff the snake has reached the food
        if self.env.snake.head.colliderect(self.env.food):
            self.env.score += 1
            self.env.snake.grow()
            self.env._place_food()

        terminated = self.env._is_collision()
        # Give a reward according to the condition
        if terminated:
            reward = 10
        elif self.env._is_collision():
            reward = -10
        else:
            reward = -1

        observation = np.linalg.norm(Line(agent_location, target_location).to_vector())
        info = {'snake_direction' : self.env.snake.direction, 'agent_location' : agent_location}

        if (self.env.render_mode is not None) and not terminated:
            self.env.render()

        return observation, reward, terminated, False, info

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
    env = Snake2dEnv(render_mode='human', width=width, height=height, nb_obstacles=obstacles)
    wrapped_env = PlayableSnake(env, fps=speed)

    pygame.init()
    _, info = wrapped_env.reset()
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
        _, _, terminated, _, info = wrapped_env.step(action)

        if terminated :
            # TODO : make possibility for user to replay
            print("You loose ! \nHit enter to retry or escape to quit.\n")
            # pygame.quit()
            # quit()
            wrapped_env.reset()


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