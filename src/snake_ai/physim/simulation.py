##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2022-12-13 4:35:40 pm
 # @copyright https://mit-license.org/
 #
from snake_ai.physim import ConvolutionWindow, DiffusionProcess, GradientField
from snake_ai.envs import SnakeClassicEnv
import matplotlib.pyplot as plt
import jax.scipy as jsp
import argparse
import logging
import time
import jax

CONV_SIZE = 10

def main(nb_obstacles=10, nb_particles=1_000, t_max=100, diff_coef=10, seed=0):
    draw = nb_particles <= 1_000
    env = SnakeClassicEnv(nb_obstacles=nb_obstacles)
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
    concentration_map = diff_process.concentration_map.T
    toc = time.perf_counter()
    print(f"Getting concentration map took {toc - tic:0.4f}s")


    conv_window = ConvolutionWindow.gaussian(CONV_SIZE)
    tic = time.perf_counter()
    smoothed_field = GradientField.smooth_field(concentration_map, conv_window)
    toc = time.perf_counter()
    print(f"Getting concentration field took {toc - tic:0.4f}s")

    log_concentration = GradientField.compute_log(smoothed_field)
    tic = time.perf_counter()
    log_grad = GradientField.compute_gradient(log_concentration)
    toc = time.perf_counter()
    print(f"Computing gradient of the concentration field took {toc - tic:0.4f}s")

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    cax = ax[0].imshow(concentration_map, cmap='inferno', interpolation='none')
    ax[0].set(title = "Concentration map")
    fig.colorbar(cax)

    cax = ax[1].imshow(smoothed_field, cmap='inferno', interpolation='none')
    ax[1].set(title = "Smoothed field")
    fig.colorbar(cax)

    cax = ax[2].imshow(log_concentration, cmap='inferno', interpolation='none')
    ax[2].quiver(log_grad[0], log_grad[1], angles='xy', scale_units='xy', scale=1)
    ax[2].set(title = "Log(Concentration) with gradients")
    fig.colorbar(cax)
    fig.tight_layout()
    # grad = GradientField.compute_gradient(smoothed_field)

    # fig2, ax2 = plt.subplots(2, 2)
    # ax2[0,0].imshow(grad[0], cmap='inferno')
    # # ax2[0, 0].set(title = "$$\nabla(c)_x$$")
    # ax2[0,1].imshow(grad[1], cmap='inferno')
    # # ax2[0, 1].set(title = "$$\nabla(c)_y$$")
    # ax2[1,0].imshow(log_grad[0], cmap='inferno')
    # # ax2[1, 0].set(title = "$$\nabla(log(c))_x$$")
    # ax2[1,1].imshow(log_grad[1], cmap='inferno')
    # # ax2[1, 1].set(title = "$$\nabla(log(c))_y$$")

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Diffusion process simulation')
    parser.add_argument('-o', '--nb_obstacles', type=int, default=10, help='Number of obstacles in the environment')
    parser.add_argument('-p', '--nb_particles', type=float, default=1_000, help='Number of particules to simulate')
    parser.add_argument('-t', '--t_max', type=int, default=100, help='Maximum simulation time')
    parser.add_argument('-D', '--diff_coef', type=float, default=100, help='Diffusion coefficient of the diffusive process')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed for the simulation PRNG')
    args = parser.parse_args()
    main(**vars(args))
