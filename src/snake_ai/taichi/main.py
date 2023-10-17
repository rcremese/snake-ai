from typing import List
import taichi as ti
import numpy as np

from snake_ai.envs import (
    GridWorld,
    GridWorld3D,
    Env3DConverter,
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


# def main(args):
#     """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

#     Instead of returning the value from :func:`fib`, it prints the result to the
#     ``stdout`` in a nicely formatted message.


#     Args:
#       args (List[str]): command line parameters as list of strings
#           (for example  ``["--verbose", "42"]``).
#     """
#     args = parse_args(args)
#     setup_logging(args.loglevel)
#     _logger.debug("Starting crazy calculations...")
#     print(f"The {args.n}-th Fibonacci number is {fib(args.n)}")
#     _logger.info("Script ends here")


def get_env_parser():
    env_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Physical simulation visualisation",
        add_help=False,
    )
    # parser.add_argument(
    #     "--name", type=str, choices=ENVIRONMENT_NAMES  # , default="grid_world"
    # )
    subparser = env_parser.add_subparsers(
        # title="pos",
        dest="name",
        help="Name of the environment",
    )
    grid_world_parser = subparser.add_parser("grid_world", help="2D grid world")
    GridWorld.add_arguments(grid_world_parser)
    rand_obs_parser = subparser.add_parser("rand_obs", help="2D random obstacles")
    RandomObstaclesEnv.add_arguments(rand_obs_parser)
    maze_parser = subparser.add_parser("maze", help="2D maze")
    MazeGrid.add_arguments(maze_parser)
    slot_parser = subparser.add_parser("slot", help="2D slot")
    SlotEnv.add_arguments(slot_parser)
    rooms_parser = subparser.add_parser("rooms", help="2D room escape")
    RoomEscape.add_arguments(rooms_parser)
    grid_world_3d_parser = subparser.add_parser("grid_world_3d", help="3D grid world")
    GridWorld3D.add_arguments(grid_world_3d_parser)
    rand_obs_3d_parser = subparser.add_parser("rand_obs_3d", help="3D random obstacles")
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
        nargs="+",
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


def run(args: List[str]):
    simulation_parser = get_simulation_parser()
    env_parser = get_env_parser()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[simulation_parser, env_parser],
        description="Physical simulation visualisation",
        add_help=True,
    )
    # parser.add_argument(
    #     "--name", type=str, choices=ENVIRONMENT_NAMES  # , default="grid_world"
    # )
    # subparser = env_parser.add_subparsers(
    #     # title="pos",
    #     dest="name",
    #     help="Name of the environment",
    # )
    # grid_world_parser = subparser.add_parser("grid_world", help="2D grid world")
    # GridWorld.add_arguments(grid_world_parser)
    # rand_obs_parser = subparser.add_parser("rand_obs", help="2D random obstacles")
    # RandomObstaclesEnv.add_arguments(rand_obs_parser)
    # maze_parser = subparser.add_parser("maze", help="2D maze")
    # MazeGrid.add_arguments(maze_parser)
    # slot_parser = subparser.add_parser("slot", help="2D slot")
    # SlotEnv.add_arguments(slot_parser)
    # rooms_parser = subparser.add_parser("rooms", help="2D room escape")
    # RoomEscape.add_arguments(rooms_parser)
    # grid_world_3d_parser = subparser.add_parser("grid_world_3d", help="3D grid world")
    # GridWorld3D.add_arguments(grid_world_3d_parser)
    # rand_obs_3d_parser = subparser.add_parser("rand_obs_3d", help="3D random obstacles")
    # RandomObstacles3D.add_arguments(rand_obs_3d_parser)

    print(parser.parse_args(args))
    return parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[env_parser, simulation_parser],
        description="Physical simulation visualisation",
    )

    # namespace_temp, _ = parser.parse_known_args(args)
    # if namespace_temp.name == "grid_world":
    #     GridWorld.add_arguments(parser)
    # elif namespace_temp.name == "grid_world_3d":
    #     GridWorld3D.add_arguments(parser)
    # elif namespace_temp.name == "rand_obs":
    #     RandomObstaclesEnv.add_arguments(parser)
    # elif namespace_temp.name == "rand_obs_3d":
    #     RandomObstacles3D.add_arguments(parser)
    # elif namespace_temp.name == "maze":
    #     MazeGrid.add_arguments(parser)
    # elif namespace_temp.name == "slot":
    #     SlotEnv.add_arguments(parser)
    # elif namespace_temp.name == "rooms":
    #     RoomEscape.add_arguments(parser)
    # else:
    #     raise ValueError(
    #         f"Unknown environment name {namespace_temp.name}. \
    #             \nThe name must be one of {ENVIRONMENT_NAMES}.\nAborting..."
    #     )


def optimize():
    pass


def main():
    seed = 10
    width, height, depth = 10, 10, 10
    nb_obs, max_size = 10, 3
    resolution = None

    setup_logging(logging.INFO)
    simulation_path = (
        Path(__file__).parents[3].joinpath("simulations").resolve(strict=True)
    )

    ti.init(arch=ti.gpu)
    env = RandomObstacles3D(
        width, height, depth, seed=seed, nb_obs=nb_obs, max_size=max_size
    )
    _logger.info("Initialize the environment")
    env.reset()
    converter = Env3DConverter(env, resolution)

    dirpath = simulation_path.joinpath(env.name, f"seed_{seed}")
    field_name = (
        "concentration_" + "x".join([str(res) for res in converter.resolution]) + ".npz"
    )

    if dirpath.exists():
        _logger.info("Loading the environment and the concentration field")
        loader = FieldLoader(dirpath.joinpath(field_name))
        concentration = loader.load()
    else:
        _logger.info("Solve the diffusion equation at stationarity")
        solver = DiffusionSolver(env, resolution)
        concentration = solver.solve(shape="box")
        # write the concentration field & the environment
        _logger.info("Writing the concentration field and the environment")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            field_writer = FieldWriter(tmpdir.joinpath(field_name))
            field_writer.write(concentration)
            env_writer = EnvWriter3D(tmpdir.joinpath("environment.json"))
            env_writer.write(env)
            shutil.move(str(tmpdir), str(dirpath))

    log_concentration = log(concentration, eps=1e-6)
    # init_pos = converter.convert_free_positions_to_point_cloud(step=3)
    init_pos = converter.get_agent_position(repeats=10)
    _logger.info("Running the simulation")
    simulation = WalkerSimulationStoch3D(
        init_pos,
        potential_field=log_concentration,
        obstacles=env.obstacles,
        t_max=200,
        dt=0.1,
        diffusivity=0.01,
    )
    simulation.reset()
    simulation.run()
    # simulation.optimize(converter.get_goal_position(), max_iter=200, lr=1)
    logging.info("Plotting the simulation result")
    print("Goal position:", converter.get_goal_position())
    vis.plot_3D_trajectory(
        simulation.positions,
        converter.get_goal_position(),
        env.obstacles,
    )


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


if __name__ == "__main__":
    # parser = argument_parser(["grid_world_3d", "--help"])
    run(["--t_max", "80", "grid_world", "--width", "50"])
    # run(["grid_world_3d", "--help"])
    # run(["--help"])
