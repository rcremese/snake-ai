import taichi as ti
import numpy as np

from snake_ai.utils.io import EnvLoader
import snake_ai.utils.visualization as vis
from snake_ai.taichi.field import ScalarField, Box2D, log
from snake_ai.taichi.walk_simulation import WalkerSimulationStoch2D
from snake_ai.utils.converter import (
    convert_free_space_to_point_cloud,
    convert_obstacles_to_physical_space,
    convert_goal_position,
)

from pathlib import Path
import argparse


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
    walk_simulation()
