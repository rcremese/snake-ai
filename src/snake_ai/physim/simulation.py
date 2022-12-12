from snake_ai.physim import ConvolutionWindow, DiffusionProcess
from snake_ai.envs import SnakeClassicEnv
import matplotlib.pyplot as plt
import jax.scipy as jsp
import argparse
import logging
import time
import jax

def concentration_field(concentration_map : jax.Array, conv_window : jax.Array):
    return jsp.signal.convolve(concentration_map, conv_window, mode='same')

def main(nb_obstacles=10, nb_particles=1_000, t_max=100, diff_coef=10, seed=0):
    draw = nb_particles <= 1_000
    env = SnakeClassicEnv(nb_obstacles=nb_obstacles)
    obstacles = [{'left' : obs.left, 'right' : obs.right, 'top' : obs.top, 'bottom' : obs.bottom} for obs in env.obstacles]
    diff_process = DiffusionProcess(nb_particles=int(nb_particles), t_max=t_max, window_size=env.window_size, diff_coef=diff_coef, seed=seed)
    env.reset()
    diff_process.reset(*env.food.center)
    logging.info('Start the simultion')
    # diff_process.start_simulation(draw)
    diff_process.accelerated_simulation(obstacles)
    logging.info(f'Simultion ended at time {diff_process.time}')
    #diff_process.draw()

    logging.info('Printing the concentration maps...')
    tic = time.perf_counter()
    concentration_map = diff_process.concentration_map.T
    toc = time.perf_counter()
    print(f"Getting concentration map took {toc - tic:0.4f}s")

    fig, ax = plt.subplots(1,2)
    cax = ax[0].imshow(concentration_map, cmap='inferno', interpolation='none')
    ax[0].set(title = "Concentration map")
    fig.colorbar(cax)
    
    conv_window = ConvolutionWindow.gaussian(env.pixel_size)
    tic = time.perf_counter()
    conc_field = concentration_field(concentration_map, conv_window)
    toc = time.perf_counter()
    print(f"Getting concentration field took {toc - tic:0.4f}s")

    cax = ax[1].imshow(conc_field, cmap='inferno', interpolation='none')
    ax[1].set(title = "Concentration field")
    fig.colorbar(cax)
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
