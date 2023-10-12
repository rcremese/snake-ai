import taichi as ti
import numpy as np

from snake_ai.envs import Env3DConverter, RandomObstacles3D
from snake_ai.taichi import DiffusionSolver, ScalarField
from snake_ai.utils.io import EnvLoader
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
import argparse
import logging


def main():
    seed = 10
    width, height, depth = 10, 10, 10
    nb_obs, max_size = 10, 3
    resolution = 10

    logging.basicConfig(level=logging.INFO)

    ti.init(arch=ti.gpu)
    env = RandomObstacles3D(
        width, height, depth, seed=seed, nb_obs=nb_obs, max_size=max_size
    )
    logging.info("Initialize the environment")
    env.reset()
    converter = Env3DConverter(env, resolution)
    logging.info("Solve the diffusion equation at stationarity")
    solver = DiffusionSolver(env, resolution)
    concentration = solver.solve(shape="point")
    log_concentration = log(concentration, eps=1e-6)
    # init_pos = converter.convert_free_positions_to_point_cloud()
    init_pos = converter.get_agent_position(5)
    logging.info("Running the simulation")
    simulation = WalkerSimulationStoch3D(
        init_pos,
        potential_field=log_concentration,
        obstacles=env.obstacles,
        t_max=100,
        dt=0.1,
        diffusivity=0.1,
    )
    # simulation.reset()
    # simulation.run()
    simulation.optimize(converter.get_goal_position(), max_iter=200, lr=1)
    logging.info("Plotting the simulation result")
    print("Goal position:", converter.get_goal_position())
    vis.plot_3D_trajectory(
        simulation.positions,
        converter.get_goal_position(),
        converter.convert_obstacles_to_binary_map(),
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
    main()
