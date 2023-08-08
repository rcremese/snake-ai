# scientific lib import
from phi.jax import flow
import matplotlib.pyplot as plt
from matplotlib import animation

# project import
from snake_ai.envs import GridWorld, RandomObstaclesEnv, MazeGrid, SlotEnv, RoomEscape
from snake_ai.physim import DiffusionSimulation, maths, autodiff
from snake_ai.utils.io import SimulationLoader, SimulationWritter
import snake_ai.physim.visualization as vis

# general import
from pathlib import Path
from typing import List
import argparse
import logging
import time

ENVIRONMENT_NAMES = ["grid_world", "rand_obs", "maze", "slot", "rooms"]

def environment_parser() -> argparse.ArgumentParser:
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
    return env_parser

def simulation_parser() -> argparse.ArgumentParser:
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
        default="crank_nicolson",
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
    return diffusion_parser

def walker_parser() -> argparse.ArgumentParser:
    walk_parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    walk_parser.add_argument("-p", "--path", type=str, required=True, help="Path to the simulation directory")
    walk_parser.add_argument("-t", "--t_max", type=int, default=100, help="Maximum time for the simulation")
    walk_parser.add_argument("--dt", type=float, default=1, help="Time step for the simulation")
    walk_parser.add_argument("--eps", type=float, default=1e-6, help="Epsilon value for the log concentration computation")
    walk_parser.add_argument("--diffusivity", type=float, default=0, help="Diffusion coefficient of the walkers. If set to zeor, the walkers are deterministic")
    return walk_parser 

def simulate_diffusion():
    env_parser = environment_parser()
    diffusion_parser = simulation_parser()
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

    args = parser.parse_args()
    diffuse(args)
    # args.func(args)

def diffuse(args: argparse.Namespace):
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

    # Visualize the initial configuration of walkers
    concentration = simulation.field
    log_concentration = maths.compute_log_concentration(concentration, epsilon=1e-6)
    force_field = flow.field.spatial_gradient(log_concentration, type=flow.CenteredGrid)
    pt_cloud = simulation.point_cloud
    fig, _, _ = vis.plot_walkers_with_concentration(
        pt_cloud, log_concentration, force_field=force_field
    )
    fig.savefig(save_dir.joinpath("initial_configuration.png"))
    
# TODO : Integrer walker dans le simulateur
def simulate_walkers():
    walk_parser = walker_parser()
    args = walk_parser.parse_args()
    walk(args)

def walk(args: argparse.Namespace):
    path = Path(args.path).resolve(strict=True)

    loader = SimulationLoader(path)
    simulation = loader.load()
    
    concentration = simulation.field
    log_concentration = maths.compute_log_concentration(concentration, epsilon=args.eps)
    force_field = flow.field.spatial_gradient(log_concentration, type=flow.CenteredGrid)
    force_field = maths.clip_gradient_norm(force_field, threashold=1)
    
    
    if args.diffusivity:
        trajectories = autodiff.stochastic_walk_simulation(simulation.point_cloud, force_field, dt=args.dt, nb_iter=args.t_max, diffusivity=args.diffusivity)
    else:
        trajectories = autodiff.deterministic_walk_simulation(simulation.point_cloud, force_field, dt=args.dt, nb_iter=args.t_max)
    
    walker_type = "stochastic" if args.diffusivity else "deterministic"
    animation_name = f"{walker_type}_walkers_Tmax={args.t_max}_dt={args.dt}_D={args.diffusivity}.gif"

    vis.animate_walk_history(
        trajectories,
        log_concentration,
        output=path.joinpath(animation_name),
        force_field=force_field,
    )

def autodiff_simulation():
    walk_parser = walker_parser()
    autodiff_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
        parents=[walk_parser], 
        description="Walker simulation optimization loop using autodiff",
    )
    autodiff_parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for the gradient descent")
    autodiff_parser.add_argument("--max_epoch", type=int, default=100, help="Maximum number of epoch for the gradient descent")
    autodiff_parser.add_argument("--gradient_clip", type=float, default=1, help="Maximum norm of the gradient")
    args = autodiff_parser.parse_args()
    autodiff_sim(args)

def autodiff_sim(args : argparse.Namespace):
    from snake_ai.physim import converter
    path = Path(args.path).resolve(strict=True)

    loader = SimulationLoader(path)
    simulation = loader.load()
    
    concentration = simulation.field
    log_concentration = maths.compute_log_concentration(concentration, epsilon=args.eps)
    force_field = flow.field.spatial_gradient(log_concentration, type=flow.CenteredGrid)

    target = converter.convert_position_to_point(simulation.env.goal, simulation.env.pixel)
    obstacles = converter.convert_obstacles_to_geometry(simulation.env.obstacles, simulation.env.pixel)
    
    loss_values, force_field = autodiff.trajectory_optimization(simulation.point_cloud, force_field, target=target, obstacles=obstacles, dt=args.dt, nb_iter=args.t_max, 
                                     lr=args.lr, max_epoch=args.max_epoch, gradient_clip=args.gradient_clip, diffusivity=args.diffusivity)

    history = autodiff.stochastic_walk_simulation(simulation.point_cloud, force_field, dt=args.dt, nb_iter=args.t_max, diffusivity=args.diffusivity)
    vis.animate_walk_history(history, log_concentration, output=path.joinpath("differentiated_walkers.gif"),
                             target=target,
                             force_field=force_field)

    vis.plot_loss(loss_values, output=path.joinpath("loss_evolution.png"))
if __name__ == "__main__":
    simulate_diffusion()
    # subparsers = parser.add_subparsers(title='subcommands', required=True, dest='subparser_name')
    # # Add diffusion solver specific parser
    # diff_sim_parser = subparsers.add_parser('diffusion', parents=[env_parser, diffusion_parser], help="Simulate the diffusion equation",
    #                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # diff_sim_parser.set_defaults(func=diffusion_simulation)
    # # Add walker specific parser
    # walker_sim_parser = subparsers.add_parser('walkers', parents=[walk_parser], help="Simulate walkers in the environment",
    #                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # walker_sim_parser.set_defaults(func=walk_simulation)
    