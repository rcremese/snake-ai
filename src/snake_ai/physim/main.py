
# scientific lib import
import numpy as np
from phi.jax import flow
import matplotlib.pyplot as plt
from matplotlib import animation
# project import
from snake_ai.physim import Simulation, DiffusionProcess, RegularGrid2D, SmoothGradientField, ConvolutionWindow
from snake_ai.physim.gradient_field import compute_log, smooth_field
from snake_ai.physim.autodiff import optimization_step
from snake_ai.envs import SnakeClassicEnv
# general import
from pathlib import Path
from typing import List
import argparse
import logging
import time

def main():
    # Parent parser for the environment
    env_parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    env_parser.add_argument('-w', '--width', type=int, default=20, help='Width of the environment')
    env_parser.add_argument('-he', '--height', type=int, default=20, help='Height of the environment')
    env_parser.add_argument('-o', '--nb_obstacles', type=int, default=10, help='Number of obstacles in the environment')
    env_parser.add_argument('-s', '--seed', type=int, default=0, help='Seed for the simulation PRNG')
    env_parser.add_argument('--pixel', type=int, default=10, help='Size of a game pixel in pixel unit')
    env_parser.add_argument('--max_size', type=int, default=2, help='Maximum size of the obstacles in terms of the pixel size')
    # Parent parser for all diffusion simulations
    diffusion_parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    diffusion_parser.add_argument('-D', '--diff_coef', type=float, default=1, help='Diffusion coefficient of the diffusive process')
    diffusion_parser.add_argument('-t', '--t_max', type=int, default=100, help='Time for the end of the simulation')
    diffusion_parser.add_argument('--eps', type=float, default=1e-6, help='Threshold for the estimation of the log')
    diffusion_parser.add_argument('--use_log', action="store_true", help='Flag to indicate whether to use log(concentration) or not')

    # Define the main parser and all the subparsed values
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Physical simulation visualisation')
    subparsers = parser.add_subparsers(title='subcommands', required=True, dest='subparser_name')
    # Add diffusion process specific parser
    diff_process_parser = subparsers.add_parser('diffusion_process', parents=[env_parser, diffusion_parser], help="Diffusion process simulation",
                                                aliases=['dp'], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    diff_process_parser.add_argument('-p', '--nb_particles', type=float, default=1_000, help='Number of particules to simulate')
    diff_process_parser.add_argument('-c ', '--conv_factor', type=float, default=1, help='Size of the convolution window in terms of the pixel size')
    diff_process_parser.set_defaults(func=diffusion_process_simulation)
    # Add diffusion solver specific parser
    diff_equation_parser = subparsers.add_parser('diffusion_equation', parents=[env_parser, diffusion_parser], help="Diffusion equation simulation",
                                                 aliases=['de'], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    diff_equation_parser.set_defaults(func=diffusion_equation_solver)
    flow.math._functional
    # args = parser.parse_args(['de', '-t', '2000', '--use_log'])
    args = parser.parse_args()
    args.func(**vars(args))

def deterministic_walk(point_cloud : flow.PointCloud, gradient_field : flow.CenteredGrid, time_step : int, goal : flow.Tensor, max_iter : int, save_step : int):
    i = 0
    history = [point_cloud]
    # jited_advection = flow.math.jit_compile(flow.advect.advect)
    while i < max_iter:
        point_cloud : flow.PointCloud = flow.advect.advect(point_cloud, gradient_field, dt=time_step)
        # point_cloud : flow.PointCloud = jited_advection(point_cloud, gradient_field, time_step)
        if i % save_step:
            history.append(point_cloud)
        i +=1
    return history

def animate_walk_history(concentration : flow.CenteredGrid, history : List[flow.PointCloud]):
    fig, ax = plt.subplots()
    ax.imshow(concentration.values.numpy('y,x'))
    ax.set(xlabel='x', ylabel='y', title='Concentration map + deterministic walkers')
    anim = []
    for point_cloud in history:
        points = point_cloud.elements.center.numpy('point,vector')
        # anim.append(flow.plot(point_cloud, animate=True))
        anim.append(ax.plot(points[:,0], points[:,1], marker="o", ls="", animated=True, color='orange'))
    return animation.ArtistAnimation(fig, anim, blit=True, interval=200)

def diffusion_equation_solver(width=20, height=20, nb_obstacles=10, max_size = 2, diff_coef=1, seed=0, pixel=10, t_max=100, use_log=False, eps=1e-6, **kwargs):
    save_path = Path(__file__).parents[3].joinpath("datas").resolve(strict=True)
    simu_name = f"diffusion_Tmax={t_max}_D={diff_coef}_Nobs={nb_obstacles}_size={max_size}_Box({width * pixel},{height * pixel})_seed={seed}"
    if save_path.joinpath(simu_name).exists():
        simu = Simulation.load(save_path.joinpath(simu_name))
    else:
        simu = Simulation(width, height, nb_obstacles, max_size, pixel, seed, diff=diff_coef, t_max=t_max)
        simu.start()
        simu.write(save_path)

    if use_log:
        log_concentration = compute_log(simu.concentration, eps=eps)
        gradient = flow.field.spatial_gradient(log_concentration)
    else:
        gradient = flow.field.spatial_gradient(simu.concentration)
    ## Visualisation
    # simu.env.render("human")
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(simu.concentration.values.numpy('y,x'), cmap='viridis')
    ax[0].set(title='Concentration + gradients', xlabel='x', ylabel='y')
    np_grad = gradient.values.numpy('vector,y,x')
    ax[0].quiver(np_grad[0], np_grad[1], angles='xy', scale_units='xy', scale=1)
    ax[1].imshow(np.linalg.norm(np_grad, axis=0))
    ax[1].set(title='Gradient norm', xlabel='x', ylabel='y')
    fig.savefig(save_path.joinpath(simu_name, 'concentration.png'))
    point_cloud = simu.get_valid_point_cloud()
    ## Optimization steps
    lr = 0.1
    grad_eval = flow.math.functional_gradient(optimization_step, wrt='velocity')
    for step in range(200):
        loss, grad = grad_eval(point_cloud.points, gradient, dt=10, gamma=0)
        gradient -= lr * grad
        print(loss)

    history = deterministic_walk(point_cloud, gradient, time_step=50, max_iter=200, save_step=10, goal=None)
    anim = animate_walk_history(simu.concentration, history)
    anim.save(save_path.joinpath(simu_name, 'video.mp4'))
    plt.show()

def diffusion_process_simulation(width=20, height=20, nb_obstacles=10, nb_particles=1_000, max_size = 3, diff_coef=10, seed=0, pixel=20, conv_factor=1, t_max=None, use_log=False, **kwargs):
    draw = nb_particles <= 1_000
    # Mean time to exit a box of dimension d for a brownian motion with diffusion D : T_max = L_max ^2 / 2dD
    mean_dist = np.sqrt(width**2 + height**2) * pixel
    if t_max is None:
        t_max = int(mean_dist**2 / (diff_coef * 8))

    env = SnakeClassicEnv(render_mode="human", width=width, height=height, pixel=pixel, max_obs_size=max_size, nb_obstacles=nb_obstacles)
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

if __name__ == '__main__':
    main()