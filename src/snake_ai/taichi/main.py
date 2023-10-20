from typing import List
import taichi as ti
import numpy as np

from snake_ai.envs import (
    GridWorld,
    GridWorld3D,
    Env3DConverter,
    EnvConverter,
    RandomObstaclesEnv,
    RandomObstacles3D,
    MazeGrid,
    SlotEnv,
    RoomEscape,
)
from snake_ai.taichi import DiffusionSolver, ScalarField
from snake_ai.utils.io import (
    EnvLoader,
    FieldLoader,
    FieldWriter,
    EnvWriter,
    EnvWriter3D,
)
import snake_ai.utils.visualization as vis
from snake_ai.taichi.field import log
from snake_ai.taichi.boxes import Box2D
from snake_ai.taichi.walk_simulation import (
    WalkerSimulationStoch2D,
    WalkerSimulationStoch3D,
)
from snake_ai.envs.converter import (
    Env2DConverter,
    Env3DConverter,
    convert_free_space_to_point_cloud,
    convert_obstacles_to_physical_space,
    convert_goal_position,
)

from pathlib import Path
import argparse, tempfile, shutil
import logging
import sys

__author__ = "Robin CREMESE"
__copyright__ = "Robin CREMESE"
__license__ = "MIT"

_logger = logging.getLogger(__name__)
ENVIRONMENT_NAMES = [
    "grid_world",
    "rand_obs",
    "maze",
    "slot",
    "rooms",
    "grid_world_3d",
    "rand_obs_3d",
]


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def get_env_parser():
    env_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Physical simulation visualisation",
        add_help=False,
    )
    subparser = env_parser.add_subparsers(
        title="Selection of the environment by name",
        dest="name",
        help="Name of the environment",
    )
    grid_world_parser = subparser.add_parser(
        "grid_world", help="2D obstacle-free environment"
    )
    GridWorld.add_arguments(grid_world_parser)
    rand_obs_parser = subparser.add_parser(
        "rand_obs", help="2D environment with randomly sampled obstacles"
    )
    RandomObstaclesEnv.add_arguments(rand_obs_parser)
    maze_parser = subparser.add_parser("maze", help="2D maze generated with mazelib")
    MazeGrid.add_arguments(maze_parser)
    slot_parser = subparser.add_parser(
        "slot", help="2D environment with 2 slot between the agent and the goal"
    )
    SlotEnv.add_arguments(slot_parser)
    rooms_parser = subparser.add_parser(
        "rooms", help="2D environment with 4 rooms and 4 entries"
    )
    RoomEscape.add_arguments(rooms_parser)
    grid_world_3d_parser = subparser.add_parser(
        "grid_world_3d", help="3D obstacle-free environment"
    )
    GridWorld3D.add_arguments(grid_world_3d_parser)
    rand_obs_3d_parser = subparser.add_parser(
        "rand_obs_3d", help="3D environment with randomly sampled obstacles"
    )
    RandomObstacles3D.add_arguments(rand_obs_3d_parser)
    return env_parser


def get_simulation_parser():
    simulation_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Physical simulation visualisation",
        add_help=False,
    )
    simulation_parser.add_argument(
        "--init_value",
        type=float,
        default=1.0,
        help="Value of the concentration field",
    )
    simulation_parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Resolution of the concentration field. "
        + "If None, the resolution is the same as the environment",
    )
    simulation_parser.add_argument(
        "--t_max", type=float, default=100.0, help="Maximum time of the simulation"
    )
    simulation_parser.add_argument(
        "--dt", type=float, default=1, help="Time step of the simulation"
    )
    simulation_parser.add_argument(
        "--diffusivity",
        type=float,
        default=0,
        help="Diffusion coefficient of the walkers. "
        + "If set to zero, the walkers are deterministic",
    )
    simulation_parser.add_argument(
        "--nb_walkers",
        type=int,
        default=100,
        help="Number of walkers in the simulation",
    )
    simulation_parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Threashold value for the log concentration",
    )

    return simulation_parser


def get_environment(args: argparse.Namespace) -> GridWorld:
    if args.name == "grid_world":
        return GridWorld(
            width=args.width, height=args.height, pixel=args.pixel, seed=args.seed
        )
    if args.name == "rand_obs":
        return RandomObstaclesEnv(
            width=args.width,
            height=args.height,
            pixel=args.pixel,
            seed=args.seed,
            max_obs_size=args.max_size,
            nb_obs=args.nb_obs,
        )
    if args.name == "maze":
        return MazeGrid(
            width=args.width,
            height=args.height,
            pixel=args.pixel,
            seed=args.seed,
            generator=args.maze_generator,
        )
    if args.name == "slot":
        return SlotEnv(
            width=args.width, height=args.height, pixel=args.pixel, seed=args.seed
        )
    if args.name == "rooms":
        return RoomEscape(
            width=args.width, height=args.height, pixel=args.pixel, seed=args.seed
        )
    if args.name == "grid_world_3d":
        return GridWorld3D(
            width=args.width,
            height=args.height,
            depth=args.depth,
            seed=args.seed,
        )
    if args.name == "rand_obs_3d":
        return RandomObstacles3D(
            width=args.width,
            height=args.height,
            depth=args.depth,
            seed=args.seed,
            max_obs_size=args.max_size,
            nb_obs=args.nb_obs,
        )
    raise ValueError(
        f"Unknown environment {args.name}. Expected one of the following values : {ENVIRONMENT_NAMES}"
    )


def run(args: List[str]):
    simulation_parser = get_simulation_parser()
    env_parser = get_env_parser()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[simulation_parser, env_parser],
        description="Physical simulation visualisation",
        add_help=True,
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the simulation directory if it already exists",
    )
    args = parser.parse_args(args)

    setup_logging(args.loglevel)

    ti.init(arch=ti.gpu)
    env = get_environment(args)
    converter = EnvConverter(env, args.resolution)

    _logger.info("Initialize the environment")
    env.reset()

    simulation_path = (
        Path(__file__).parents[3].joinpath("simulations").resolve(strict=True)
    )
    dirpath = simulation_path.joinpath(env.name, f"seed_{args.seed}")
    field_name = (
        "concentration_" + "x".join([str(res) for res in converter.resolution]) + ".npz"
    )

    if dirpath.joinpath(field_name).exists():
        _logger.info("Loading the environment and the concentration field")
        loader = FieldLoader(dirpath.joinpath(field_name))
        concentration = loader.load()
    else:
        _logger.info("Solve the diffusion equation at stationarity")
        solver = DiffusionSolver(env, args.resolution)
        concentration = solver.solve(shape="box")
        # write the concentration field & the environment
        _logger.info("Writing the concentration field and the environment")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            field_writer = FieldWriter(tmpdir.joinpath(field_name))
            field_writer.write(concentration)
            if isinstance(env, GridWorld):
                env_writer = EnvWriter(tmpdir.joinpath("environment.json"))
            else:
                env_writer = EnvWriter3D(tmpdir.joinpath("environment.json"))
            env_writer.write(env)
            shutil.copytree(str(tmpdir), str(dirpath), dirs_exist_ok=True)
            shutil.rmtree(str(tmpdir))

    log_concentration = log(concentration, eps=args.eps)
    # init_pos = converter.convert_free_positions_to_point_cloud(step=3)
    init_pos = converter.get_agent_position(repeats=args.nb_walkers)
    _logger.info("Running the simulation")
    if isinstance(env, GridWorld):
        simulation = WalkerSimulationStoch2D(
            init_pos,
            log_concentration,
            obstacles=env.obstacles,
            t_max=args.t_max,
            dt=args.dt,
            diffusivity=args.diffusivity,
        )
    else:
        simulation = WalkerSimulationStoch3D(
            init_pos,
            potential_field=log_concentration,
            obstacles=env.obstacles,
            t_max=args.t_max,
            dt=args.dt,
            diffusivity=args.diffusivity,
        )
    simulation.reset()
    simulation.run()
    # simulation.optimize(converter.get_goal_position(), max_iter=200, lr=1)
    logging.info("Plotting the simulation result")
    print("Goal position:", converter.get_goal_position())
    if isinstance(env, GridWorld):
        fig = vis.plot_2D_trajectory(
            simulation.positions,
            converter.get_goal_position(),
            env.obstacles,
            concentration=log_concentration,
            title=f"Simulation of {env.name} with {args.nb_walkers} walkers",
        )
    else:
        fig = vis.plot_3D_trajectory(
            simulation.positions,
            converter.get_goal_position(),
            env.obstacles,
            title=f"Simulation of {env.name} with {args.nb_walkers} walkers",
        )
    fig.savefig(dirpath.joinpath("simulation.png"), dpi=300)


def optimize():
    pass


# def main():
#     seed = 10
#     width, height, depth = 10, 10, 10
#     nb_obs, max_size = 10, 3
#     resolution = 3

#     setup_logging(logging.INFO)
#     simulation_path = (
#         Path(__file__).parents[3].joinpath("simulations").resolve(strict=True)
#     )

#     ti.init(arch=ti.gpu)
#     env = RandomObstacles3D(
#         width, height, depth, seed=seed, nb_obs=nb_obs, max_size=max_size
#     )
#     _logger.info("Initialize the environment")
#     env.reset()
#     converter = Env3DConverter(env, resolution)

#     dirpath = simulation_path.joinpath(env.name, f"seed_{seed}")
#     field_name = (
#         "concentration_" + "x".join([str(res) for res in converter.resolution]) + ".npz"
#     )

#     if dirpath.exists():
#         _logger.info("Loading the environment and the concentration field")
#         loader = FieldLoader(dirpath.joinpath(field_name))
#         concentration = loader.load()
#     else:
#         _logger.info("Solve the diffusion equation at stationarity")
#         solver = DiffusionSolver(env, resolution)
#         concentration = solver.solve(shape="point")
#         # write the concentration field & the environment
#         _logger.info("Writing the concentration field and the environment")
#         with tempfile.TemporaryDirectory() as tmpdir:
#             tmpdir = Path(tmpdir)
#             field_writer = FieldWriter(tmpdir.joinpath(field_name))
#             field_writer.write(concentration)
#             env_writer = EnvWriter3D(tmpdir.joinpath("environment.json"))
#             env_writer.write(env)
#             shutil.move(str(tmpdir), str(dirpath))

#     log_concentration = log(concentration, eps=1e-6)
#     init_pos = converter.convert_free_positions_to_point_cloud(step=3)
#     # init_pos = converter.get_agent_position(repeats=10)
#     _logger.info("Running the simulation")
#     simulation = WalkerSimulationStoch3D(
#         init_pos,
#         potential_field=log_concentration,
#         obstacles=env.obstacles,
#         t_max=200,
#         dt=0.1,
#         diffusivity=0.01,
#     )
#     simulation.reset()
#     simulation.run()
#     # simulation.optimize(converter.get_goal_position(), max_iter=200, lr=1)
#     logging.info("Plotting the simulation result")
#     print("Goal position:", converter.get_goal_position())
#     vis.plot_3D_trajectory(
#         simulation.positions,
#         converter.get_goal_position(),
#         env.obstacles,
#     )


def walker_parser() -> argparse.ArgumentParser:
    walk_parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    walk_parser.add_argument(
        "-p", "--path", type=str, required=True, help="Path to the simulation directory"
    )
    walk_parser.add_argument(
        "-t", "--t_max", type=int, default=100, help="Maximum time for the simulation"
    )
    walk_parser.add_argument(
        "--dt", type=float, default=1, help="Time step for the simulation"
    )
    walk_parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Epsilon value for the log concentration computation",
    )
    walk_parser.add_argument(
        "--diffusivity",
        type=float,
        default=0,
        help="Diffusion coefficient of the walkers. If set to zeor, the walkers are deterministic",
    )
    return walk_parser


def walk(args: argparse.Namespace):
    path = Path(args.path).resolve(strict=True)
    if not path.joinpath("environment.json").exists():
        raise FileNotFoundError(
            "The given path does not contain an environment.json file"
        )
    loader = EnvLoader(path.joinpath("environment.json"))
    env = loader.load()
    env.reset()

    if not path.joinpath("field.npz").exists():
        raise FileNotFoundError("The given path does not contain a field.npz file")
    object = np.load(path.joinpath("field.npz"))

    ti.init(arch=ti.gpu)
    bounds = Box2D(object["lower"], object["upper"])
    concentration = ScalarField(object["data"], bounds)
    log_concentration = log(concentration, args.eps)

    starting_pos = convert_free_space_to_point_cloud(env)
    simulation = WalkerSimulationStoch2D(
        starting_pos,
        log_concentration,
        obstacles=convert_obstacles_to_physical_space(env),
        t_max=args.t_max,
        diffusivity=args.diffusivity,
    )
    simulation.reset()
    print("Running simulation...")
    simulation.run()

    print("Saving initial walkers...")
    vis.animate_walk_history(
        simulation.positions,
        log_concentration.values.to_numpy(),
        bound_limits=object["upper"],
        output_path=path.joinpath("initial_walkers.gif"),
    )

    print("Optimizing walkers...")
    simulation.optimize(convert_goal_position(env), max_iter=200, lr=1)

    print("Saving final walkers...")
    vis.animate_walk_history(
        simulation.positions,
        log_concentration.values.to_numpy(),
        bound_limits=object["upper"],
        output_path=path.joinpath("optimized_walkers.gif"),
    )


def walk_simulation():
    walk_parser = walker_parser()
    args = walk_parser.parse_args()
    walk(args)


def main():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    run(sys.argv[1:])


if __name__ == "__main__":
    # parser = argument_parser(["grid_world_3d", "--help"])
    run(["--res", "3", "rand_obs"])
    # run(["--t_max", "80", "grid_world", "--width", "50"])
    # run(["grid_world_3d", "--help"])
    # run(["--help"])
