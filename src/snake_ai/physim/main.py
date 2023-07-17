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
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def simulate():
    # Parent parser for the environment
    env_parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    env_parser.add_argument(
        "name",
        type=str,
        choices=ENVIRONMENT_NAMES,
        help="Name of the environment to simulate",
    )
    env_parser.add_argument(
        "--width", type=int, default=20, help="Width of the environment"
    )
    env_parser.add_argument(
        "--height", type=int, default=20, help="Height of the environment"
    )
    env_parser.add_argument(
        "--pixel", type=int, default=10, help="Size of a game pixel in pixel unit"
    )
    env_parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the simulation PRNG"
    )

    # Add specific parser for each environment
    namespace_temp, _ = env_parser.parse_known_args()
    if namespace_temp.name == "rand_obs":
        env_parser.add_argument(
            "--max_size",
            type=int,
            default=3,
            help="Maximum size of the obstacles in terms of the pixel size",
        )
        env_parser.add_argument(
            "--nb_obs",
            type=int,
            default=10,
            help="Number of obstacles in the environment",
        )
    elif namespace_temp.name == "maze":
        env_parser.add_argument(
            "--maze_generator",
            type=str,
            default="prims",
            choices=MazeGrid.maze_generator,
            help="Algorithm used for maze generation",
        )

    # Parent parser for all diffusion simulations
    diffusion_parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    diffusion_parser.add_argument(
        "-D", "--diff_coef", type=float, default=1, help="Diffusion coefficient"
    )
    diffusion_parser.add_argument(
        "-T",
        "--t_max",
        type=int,
        default=None,
        help="Time limit for the end of the simulation",
    )
    diffusion_parser.add_argument(
        "--dt", type=float, default=None, help="Time step for the simulation"
    )
    diffusion_parser.add_argument(
        "--res",
        type=str,
        default="meta",
        choices=DiffusionSimulation.resolutions,
        help="Resolution used for the simulation",
    )
    diffusion_parser.add_argument(
        "--solver",
        type=str,
        default="explicit",
        choices=DiffusionSimulation.solvers,
        help="Name of the solver used for the simulation",
    )
    diffusion_parser.add_argument(
        "--stationary",
        action="store_true",
        help="Flag to indicate if the diffusion process is stationary or not",
    )
    diffusion_parser.add_argument(
        "--init_value",
        type=float,
        default=1,
        help="Value of the initial concentration field",
    )

    # Define the main parser and all the subparsed values
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[env_parser, diffusion_parser],
        description="Physical simulation visualisation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Flag to indicate if the simulation should be overwritten or not",
    )
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
        env = GridWorld(
            width=args.width, height=args.height, pixel=args.pixel, seed=args.seed
        )
    elif args.name == "rand_obs":
        env = RandomObstaclesEnv(
            width=args.width,
            height=args.height,
            pixel=args.pixel,
            seed=args.seed,
            max_obs_size=args.max_size,
            nb_obs=args.nb_obs,
        )
    elif args.name == "maze":
        env = MazeGrid(
            width=args.width,
            height=args.height,
            pixel=args.pixel,
            seed=args.seed,
            generator=args.maze_generator,
        )
    elif args.name == "slot":
        env = SlotEnv(
            width=args.width, height=args.height, pixel=args.pixel, seed=args.seed
        )
    elif args.name == "rooms":
        env = RoomEscape(
            width=args.width, height=args.height, pixel=args.pixel, seed=args.seed
        )
    else:
        raise ValueError(
            f"Unknown environment {args.name}. Expected one of the following values : {ENVIRONMENT_NAMES}"
        )
    # Define a simulation with the input parameters and execute it
    simulation = DiffusionSimulation(
        env,
        diffusivity=args.diff_coef,
        t_max=args.t_max,
        dt=args.dt,
        res=args.res,
        solver=args.solver,
        stationary=args.stationary,
        init_value=args.init_value,
    )

    # Check if the simulation already exists
    simulation_dir = (
        Path(__file__).parents[3].joinpath("simulations").resolve(strict=True)
    )
    save_dir = simulation_dir.joinpath(simulation.name, f"seed_{args.seed}")
    if save_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"The simulation {simulation.name} already exists. Use the --overwrite flag to overwrite it."
        )
    sim_writter = SimulationWritter(save_dir)
    # Start the simulation and save the result
    simulation.reset(args.seed)
    simulation.start()
    sim_writter.write(simulation)

if __name__ == "__main__":
    simulate()
