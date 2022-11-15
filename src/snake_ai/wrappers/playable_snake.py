import argparse
import pygame
from snake_ai.envs.snake_2d_env import Snake2dEnv, BLACK, WHITE, GREEN, RED, FONT_PATH
from snake_ai.utils.line import Line

import gym
import numpy as np
from typing import Tuple, List

from snake_ai.envs.utils import Direction, Reward

class PlayableSnake(gym.Wrapper):
    def __init__(self, env: Snake2dEnv, fps : int = 10):
        assert isinstance(env, Snake2dEnv)
        super().__init__(env)
        self.env.metadata['render_fps'] = fps
        self.observation_space = gym.spaces.Box(low=0, high=env.max_dist, shape=(1,))
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, **kwargs) -> Tuple[float, dict]:
        super().reset(**kwargs)
        info = self.env.info
        obs =  np.linalg.norm(Line(info['snake_head'], info['food']).to_vector())
        return obs

    def step(self, action: int) -> Tuple[np.array, float, bool, bool, dict]:
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self.direction_from_action(action)
        self.env.snake.move(direction)
        # direction = self._action_to_direction[action]
        agent_location = self.env.snake.head.center
        target_location = self.env._food.center
        # An episode is done iff the snake has reached the food
        if self.env.snake.head.colliderect(self.env._food):
            self.env.score += 1
            self.env.snake.grow()
            self.env._place_food()

        terminated = self.env._is_collision()
        # Give a reward according to the condition
        if terminated:
            reward = Reward.FOOD
        elif self.env._is_collision():
            reward = Reward.COLLISION
        else:
            reward = Reward.COLLISION_FREE

        observation = np.linalg.norm(Line(agent_location, target_location).to_vector())
        info = {'snake_direction' : self.env.snake.direction, 'agent_location' : agent_location}

        # if (self.env.render_mode is not None) and not terminated:
        #     self.env.render()

        return observation, reward, terminated, info

    def render(self, mode="human", **kwargs):
        if self.window is None:
            self.window = pygame.display.set_mode(self.env.window_size)
        if self.font is None:
            self.font = pygame.font.Font(FONT_PATH, 25)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.env.window_size)
        canvas.fill(BLACK)

        # Draw snake
        self.env.snake.draw(canvas)
        # Draw obstacles
        for obstacle in self.env.obstacles:
            pygame.draw.rect(canvas, RED, obstacle)

        # Draw food
        pygame.draw.rect(canvas, GREEN, self.env._food)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            # Draw the text showing the score
            text = self.font.render(f"Score: {self.score}", True, WHITE)
            self.window.blit(text, [0, 0])
            # update only the snake and the food
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

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