
# scientific lib import
import numpy as np
from phi.jax import flow
import matplotlib.pyplot as plt
from matplotlib import animation
# project import
from snake_ai.envs import GridWorld, RandomObstaclesEnv, MazeGrid, SlotEnv, RoomEscape
from snake_ai.physim import DiffusionSimulation
from snake_ai.utils.io import SimulationLoader, SimulationWritter
from snake_ai.physim.autodiff import optimization_step
# general import
from pathlib import Path
from typing import List
import argparse
import logging
import time

ENVIRONMENT_NAMES = ["grid_world", "rand_obs", "maze", "slot", "rooms"]

def simulate():
    # Parent parser for the environment
    env_parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    env_parser.add_argument('name', type=str, choices=ENVIRONMENT_NAMES, help='Name of the environment to simulate')
    env_parser.add_argument('--width', type=int, default=20, help='Width of the environment')
    env_parser.add_argument('--height', type=int, default=20, help='Height of the environment')
    env_parser.add_argument('--pixel', type=int, default=10, help='Size of a game pixel in pixel unit')
    env_parser.add_argument('--seed', type=int, default=0, help='Seed for the simulation PRNG')

    # Add specific parser for each environment
    namespace_temp, _ = env_parser.parse_known_args()
    if namespace_temp.name == "rand_obs":
        env_parser.add_argument('--max_size', type=int, default=3, help='Maximum size of the obstacles in terms of the pixel size')
        env_parser.add_argument('--nb_obs', type=int, default=10, help='Number of obstacles in the environment')
    elif namespace_temp.name == "maze":
        env_parser.add_argument('--maze_generator', type=str, default="prims", choices=MazeGrid.maze_generator, help='Algorithm used for maze generation')

    # Parent parser for all diffusion simulations
    diffusion_parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    diffusion_parser.add_argument('-D', '--diff_coef', type=float, default=1, help='Diffusion coefficient')
    diffusion_parser.add_argument('-T', '--t_max', type=int, default=None, help='Time limit for the end of the simulation')
    diffusion_parser.add_argument('--dt', type=float, default=None, help='Time step for the simulation')
    diffusion_parser.add_argument('--res', type=str, default="meta", choices=DiffusionSimulation.resolutions, help='Resolution used for the simulation')
    diffusion_parser.add_argument('--solver', type=str, default="crank_nicolson", choices=DiffusionSimulation.solvers, help='Name of the solver used for the simulation')
    diffusion_parser.add_argument('--stationary', action="store_true", help='Flag to indicate if the diffusion process is stationary or not')
    diffusion_parser.add_argument('--init_value', type=float, default=1, help='Value of the initial concentration field')

    # Define the main parser and all the subparsed values
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[env_parser, diffusion_parser],
                                      description='Physical simulation visualisation')
    parser.add_argument('--overwrite', action="store_true", help='Flag to indicate if the simulation should be overwritten or not')
    # subparsers = parser.add_subparsers(title='subcommands', required=True, dest='subparser_name')
    # # Add diffusion solver specific parser
    # grid_parser = subparsers.add_parser('grid_world', parents=[env_parser, diffusion_parser], help="Diffusion equation in a grid world",
    #                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # grid_parser.set_defaults(env=GridWorld(**vars(grid_parser.parse_args())))
    # args = parser.parse_args(['de', '-t', '2000', '--use_log'])
    args = parser.parse_args()
    diffusion_simulation(args)

def diffusion_simulation(args: argparse.Namespace):
    if args.name == "grid_world":
        env = GridWorld(width=args.width, height=args.height, pixel=args.pixel, seed=args.seed)
    elif args.name == "rand_obs":
        env = RandomObstaclesEnv(width=args.width, height=args.height, pixel=args.pixel, seed=args.seed, max_obs_size=args.max_size, nb_obs=args.nb_obs)
    elif args.name == "maze":
        env = MazeGrid(width=args.width, height=args.height, pixel=args.pixel, seed=args.seed, generator=args.maze_generator)
    elif args.name == "slot":
        env = SlotEnv(width=args.width, height=args.height, pixel=args.pixel, seed=args.seed)
    elif args.name == "rooms":
        env = RoomEscape(width=args.width, height=args.height, pixel=args.pixel, seed=args.seed)
    else:
        raise ValueError(f"Unknown environment {args.name}. Expected one of the following values : {ENVIRONMENT_NAMES}")
    # Define a simulation with the input parameters and execute it
    simulation = DiffusionSimulation(env, diffusivity=args.diff_coef, t_max=args.t_max, dt=args.dt, res=args.res, solver=args.solver,
                                     stationary=args.stationary, init_value=args.init_value)

    # Check if the simulation already exists
    simulation_dir = Path(__file__).parents[3].joinpath("simulations").resolve(strict=True)
    save_dir = simulation_dir.joinpath(simulation.name, f"seed_{args.seed}")
    if save_dir.exists() and not args.overwrite:
        raise FileExistsError(f"The simulation {simulation.name} already exists. Use the --overwrite flag to overwrite it.")
    sim_writter = SimulationWritter(save_dir)
    # Start the simulation and save the result
    simulation.reset(args.seed)
    simulation.start()
    sim_writter.write(simulation)

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

def compute_log(field : flow.CenteredGrid, eps : float = 1e-6):
    threashold = flow.math.where(field.values < eps, eps, field.values)
    return flow.CenteredGrid(flow.math.log(threashold), extrapolation=np.log(eps), bounds=field.bounds, resolution=field.resolution)

def diffusion_equation_solver(width=20, height=20, nb_obstacles=10, max_size = 2, diff_coef=1, seed=0, pixel=10, t_max=100, use_log=False, eps=1e-6, **kwargs):
    save_path = Path(__file__).parents[3].joinpath("datas").resolve(strict=True)
    simu_name = f"diffusion_Tmax={t_max}_D={diff_coef}_Nobs={nb_obstacles}_size={max_size}_Box({width * pixel},{height * pixel})_seed={seed}"
    if save_path.joinpath(simu_name).exists():
        loader = SimulationLoader(save_path.joinpath(simu_name))
        simu = loader.load()
    else:
        simu = DiffusionSimulation(width, height, nb_obstacles, max_size, pixel, seed, diff=diff_coef, t_max=t_max)
        simu.start()
        writer = SimulationWritter(save_path.joinpath(simu_name))
        writer.write(simu)

    if use_log:
        log_concentration = compute_log(simu.field, eps=eps)
        gradient = flow.field.spatial_gradient(log_concentration)
    else:
        gradient = flow.field.spatial_gradient(simu.field)
    ## Visualisation
    # simu.env.render("human")
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(simu.field.values.numpy('y,x'), cmap='viridis')
    ax[0].set(title='Concentration + gradients', xlabel='x', ylabel='y')
    np_grad = gradient.values.numpy('vector,y,x')
    ax[0].quiver(np_grad[0], np_grad[1], angles='xy', scale_units='xy', scale=1)
    ax[1].imshow(np.linalg.norm(np_grad, axis=0))
    ax[1].set(title='Gradient norm', xlabel='x', ylabel='y')
    fig.savefig(save_path.joinpath(simu_name, 'concentration.png'))
    point_cloud = simu.point_cloud
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


if __name__ == '__main__':
    simulate()