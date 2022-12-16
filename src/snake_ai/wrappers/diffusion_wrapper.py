##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-11-02 2:10:17 pm
 # @copyright https://mit-license.org/
 #

# TODO : Rename and rewrite the class to correspond to new case
import gym
import gym.spaces
import pygame
import numpy as np
import jax.scipy as jsp

from abc import abstractmethod, ABCMeta
import logging

from snake_ai.envs import SnakeClassicEnv
from snake_ai.physim import DiffusionProcess, ConvolutionWindow
class BaseDiffusionWrapper(gym.Wrapper, metaclass=ABCMeta):
    def __init__(self, env: SnakeClassicEnv, diffusion_coef : float = 1, seed : int = 0):
        # if not isinstance(env, SnakeClassicEnv):
        #     raise TypeError(f"Supported environment are SnakeClassicEnv, not {type(env)}")

        super().__init__(env)
        self.env : SnakeClassicEnv
        self.observation_space = gym.spaces.Box(low=np.zeros(3), high=np.ones(3), shape=(3,))

        if diffusion_coef < 0:
            raise ValueError(f"Diffusion coeffient need to be positive, not {diffusion_coef}")
        self._diffusion_coef = diffusion_coef

        if not isinstance(seed, int):
            raise TypeError(f"The seed need to be an integer, not {type(seed)}")
        self._seed = seed
        self.env.seed(seed)

        self._diffusive_field = None

    def reset(self):
        self.env.reset()
        self._diffusive_field = self._get_diffusion_field()
        return self._get_obs()

    def step(self, action):
        _, reward, done, info = super().step(action)
        if info['truncated']:
            self._diffusive_field = self._get_diffusion_field()
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert self._diffusive_field is not None, "No diffusive field computed"
        observation = np.zeros(3)
        neighbours = self.env._get_neighbours()
        for i, neighbour in enumerate(neighbours):
            # observation is 0 if the neighbour bounding box collide with obtacles, snake body or is outside
            if self.env._is_collision(neighbour):
                continue
            window = [*neighbour.topleft, *neighbour.bottomright]
            observation[i] = np.mean(self._diffusive_field[window[0]:window[2], window[1]:window[3]])
        return observation

    def render(self, mode="human", **kwargs):
        # fill canvas with the normalized diffusion field
        if self._diffusive_field is None:
            self._diffusive_field = self._get_diffusion_field()
        surf = np.zeros((*self.env.window_size, 3))
        surf[:,:,1] = 255 * self._diffusive_field # fill only the green part
        canvas = pygame.surfarray.make_surface(surf)

        return self.env.render(mode, canvas)

    @abstractmethod
    def _get_diffusion_field(self) -> np.ndarray:
        raise NotImplementedError

class DeterministicDiffusionWrapper(BaseDiffusionWrapper):
    def _get_diffusion_field(self) -> np.ndarray:
        x,y = np.meshgrid(range(self.env.window_size[0]), range(self.env.window_size[0]))
        mu = self.env._food.center
        diffusive_field = np.exp(- 0.5 / self._diffusion_coef *((x - mu[1])**2 + (y - mu[0])**2))
        assert np.max(diffusive_field) != np.min(diffusive_field), "Diffion field is constant"
        return (diffusive_field - np.min(diffusive_field)) / (np.max(diffusive_field) - np.min(diffusive_field))

class StochasticDiffusionWrapper(BaseDiffusionWrapper):
    def __init__(self, env: SnakeClassicEnv, diffusion_coef: float = 1, nb_part : float = 1e6, t_max : int = 100, seed : int = 0):
        super().__init__(env, diffusion_coef, seed)
        self._diffusion_process = DiffusionProcess(nb_part, t_max=t_max, window_size=self.env.window_size, diff_coef=self._diffusion_coef, obstacles=self.env.obstacles, seed=seed)

    def _get_diffusion_field(self) -> np.ndarray:
        logging.info('Initialisation of the diffusion process.')
        self._diffusion_process.reset(*self.env.food.center)
        logging.info('Begining of the simulation...')
        self._diffusion_process.start_simulation()
        logging.info('Computing the concentration map...')
        concentration_map = self._diffusion_process.concentration_map
        conv_window = ConvolutionWindow.gaussian(self.env.pixel_size)
        diffusive_field = jsp.signal.convolve(concentration_map, conv_window, mode='same')

        assert np.max(diffusive_field) != np.min(diffusive_field), "Diffion field is constant"
        return (diffusive_field - np.min(diffusive_field)) / (np.max(diffusive_field) - np.min(diffusive_field))
