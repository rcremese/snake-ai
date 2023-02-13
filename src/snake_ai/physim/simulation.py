##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-12-13 4:35:40 pm
 # @copyright https://mit-license.org/
 #
from snake_ai.physim import ConvolutionWindow, DiffusionProcess, Walker, RegularGrid2D, DiffusionSolver2D
from snake_ai.physim.gradient_field import SmoothGradientField, smooth_field, compute_log
from snake_ai.envs import SnakeClassicEnv
import matplotlib.pyplot as plt
from phi.jax import flow
from pathlib import Path
import numpy as np
import argparse
import logging
import time
import pygame

class Simulation:
    env : SnakeClassicEnv
    diffusion_solver : DiffusionSolver2D
    diff : float
    t_max : float
    _seed : int
    _name : str

    def __init__(self, width, height, pixel, nb_obs, diff, seed, t_max) -> None:
        # Initialise the environment and seed it
        # TODO : replace numpy random with jax random to control the seed
        self.env = SnakeClassicEnv(render_mode=None, width=width, height=height, nb_obstacles=nb_obs, pixel=pixel)
        if not isinstance(seed, int):
            raise TypeError(f"Seed need to be an int. Get {type(seed)}")
        self._seed = seed
        self.env.seed(self._seed)

        if t_max <= 0 or diff <= 0:
            raise ValueError(f"The diffusion coefficient and the maximum simulation time need to be > 0. Get {diff} and {t_max} instead")
        self.t_max = t_max
        self.diff = diff
        # Arguments to be instanciated
        self.diffusion_solver = None

    def reset(self):
        self.env.reset()
        x_max, y_max = self.env.window_size
        self.diffusion_solver = DiffusionSolver2D(x_max, y_max, self.t_max, source=self.env.food, obstacles=self.env.obstacles, diff_coef=self.diff)

    def load(self, dirpath : str or Path):
        dirpath = Path(dirpath).resolve(strict=True)

    def write(self, dirpath : str or Path):
        pass

    def get_gradient(self, field : flow.Grid) -> flow.Grid:
        flow.math.spatial_gradient()

    def start_simulation(self):
        if self.diffusion_solver is None:
            self.reset()
        self.diffusion_solver.start()

    @property
    def concentration(self):
        "Concentration field at time Tmax in the given environment"
        if self.diffusion_solver is None:
            self.reset()
        return self.diffusion_solver.concentration

    def __repr__(self) -> str:
        return f"{__class__.__name__}(env={self.env!r}, solver={self.diffusion_solver!r})"

def main():
    parser = argparse.ArgumentParser('Diffusion process simulation')
    parser.add_argument('-w', '--width', type=int, default=20, help='Width of the environment')
    parser.add_argument('-he', '--height', type=int, default=20, help='Height of the environment')
    parser.add_argument('-o', '--nb_obstacles', type=int, default=10, help='Number of obstacles in the environment')
    parser.add_argument('-p', '--nb_particles', type=float, default=1_000, help='Number of particules to simulate')
    parser.add_argument('-D', '--diff_coef', type=float, default=10, help='Diffusion coefficient of the diffusive process')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed for the simulation PRNG')
    parser.add_argument('-t', '--t_max', type=int, default=None, help='Time for the end of the simulation')
    parser.add_argument('-c ', '--conv_factor', type=float, default=1, help='Size of the convolution window in terms of the pixel size')
    parser.add_argument('--use_log', action="store_true", help='Flag to indicate whether to use log(concentration) or not')
    parser.add_argument('--pixel', type=int, default=20, help='Size of a game pixel in pixel unit')
    args = parser.parse_args()
    diffusion_process_simulation(**vars(args))

def diffusion_process_simulation(width=20, height=20, nb_obstacles=10, nb_particles=1_000, diff_coef=10, seed=0, pixel=20, conv_factor=1, t_max=None, use_log=False):
    draw = nb_particles <= 1_000
    # Mean time to exit a box of dimension d for a brownian motion with diffusion D : T_max = L_max ^2 / 2dD
    mean_dist = np.sqrt(width**2 + height**2) * pixel
    if t_max is None:
        t_max = int(mean_dist**2 / (diff_coef * 8))

    env = SnakeClassicEnv(render_mode="human", width=width, height=height, pixel=pixel, nb_obstacles=nb_obstacles)
    env.seed(seed)
    env.reset()
    diff_process = DiffusionProcess(nb_particles=int(nb_particles), t_max=t_max, window_size=env.window_size, diff_coef=diff_coef, obstacles=env.obstacles, seed=seed)
    diff_process.reset(*env.food.center)

    logging.info('Start the simultion')
    diff_process.draw()
    diff_process.start_simulation(draw)
    logging.info(f'Simultion ended at time {diff_process.time}')
    diff_process.draw()

    logging.info('Printing the concentration maps...')
    tic = time.perf_counter()
    concentration_map = compute_log(diff_process.concentration_map.T) if use_log else  diff_process.concentration_map.T
    toc = time.perf_counter()
    print(f"Getting concentration map took {toc - tic:0.4f}s")
    conv_mode = 'valid' if use_log else 'same'

    conv_size = int(conv_factor * env.pixel_size)
    grid_2d = RegularGrid2D.unitary_rectangle(*env.window_size)
    smoothed_grad = SmoothGradientField(concentration_map, grid_2d, conv_size=conv_size, conv_mode=conv_mode)

    tic = time.perf_counter()
    smoothed_field = smooth_field(concentration_map, ConvolutionWindow.gaussian(conv_size), mode=conv_mode)
    toc = time.perf_counter()
    print(f"Getting concentration field took {toc - tic:0.4f}s")

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    cax = ax[0,0].imshow(concentration_map, cmap='inferno', interpolation='none')
    ax[0, 0].set(title = 'log(concentration map)' if use_log else 'Concentration map')
    fig.colorbar(cax)

    cax = ax[0,1].imshow(smoothed_field, cmap='inferno', interpolation='none')
    ax[0, 1].quiver(smoothed_grad.dx, smoothed_grad.dy, angles='xy', scale_units='xy', scale=1)
    ax[0,1].set(title = "Smoothed field with gradients")
    fig.colorbar(cax)

    cax = ax[1,0].imshow(np.linalg.norm(np.stack([smoothed_grad.dx, smoothed_grad.dy]), axis=0), cmap='inferno', interpolation='none')
    ax[1,0].set(title = "Gradient norm")
    fig.colorbar(cax)

    cax = ax[1,1].imshow(np.arctan2(smoothed_grad.dy, smoothed_grad.dx), cmap='inferno', interpolation='none')
    ax[1,1].set(title = "arctan2(\delta_x field, \delta_y field)")
    fig.colorbar(cax)

    fig.tight_layout()
    plt.show()

def draw_walker(snake_env : SnakeClassicEnv, walker : Walker, smoothed_field : np.array):
    regular_smoothed_field = (smoothed_field.T - np.min(smoothed_field)) / (np.max(smoothed_field) - np.min(smoothed_field))
    surf = np.zeros((*regular_smoothed_field.shape, 3))
    surf[:,:,1] = 255 * regular_smoothed_field# fill only the green part

    canvas = pygame.surfarray.make_surface(surf)
    walker.draw(canvas)
    snake_env.render("human", canvas)

if __name__ == '__main__':
    main()