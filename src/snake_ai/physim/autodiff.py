##
# @author  <robin.cremese@gmail.com>
# @file Description
# @desc Created on 2023-06-24 10:07:34 am
# @copyright MIT License
#
from snake_ai.envs import RoomEscape, SlotEnv, RandomObstaclesEnv
from snake_ai.physim.simulation import DiffusionSimulation
from snake_ai.physim.solver import DiffusionSolver
from snake_ai.physim import maths
from snake_ai.utils.io import SimulationWritter, SimulationLoader
from phi.jax import flow
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

MAX_ITER = 100
MAX_EPOCH = 50


@partial(flow.math.jit_compile, auxiliary_args="dt,nb_iter")
def deterministic_walk_simulation(
    point_cloud: flow.PointCloud,
    force_field: flow.CenteredGrid,
    dt: float,
    nb_iter: int = MAX_ITER,
) -> flow.PointCloud:
    nb_iter = int(nb_iter)
    history = nb_iter * [0]
    history[0] = point_cloud
    for i in range(1, nb_iter):
        point_cloud = flow.advect.advect(point_cloud, force_field, dt=dt)
        history[i] = point_cloud
    return flow.field.stack(history, flow.batch("time"))

@partial(flow.functional_gradient, wrt="force_field")
def simulation(
    point_cloud: flow.PointCloud,
    force_field: flow.CenteredGrid,
    target: flow.Tensor,
    dt: float,
    nb_iter: int = MAX_ITER,
):
    trajectories = deterministic_walk_simulation(
        point_cloud, force_field, dt=dt, nb_iter=nb_iter
    )
    return flow.math.mean(maths.normalized_l2_distance(trajectories, target))

def close_to_target(point_cloud: flow.PointCloud, force_field : flow.CenteredGrid, target: flow.Tensor, threashold: float = 1):
    target = flow.vec(x = 15.5, y=5.5)
    loss = []
    for e in range(MAX_EPOCH):
        diff_simulation = flow.functional_gradient(simulation, wrt="force_field")
        value, grad = diff_simulation(point_cloud, force_field, target, dt=1, nb_iter=20)
        print(f"iter : {e}, loss : {value}")
        loss.append(value.numpy())
        force_field -= grad
        force_field = maths.clip_gradient_norm(force_field, threashold=1)
    return loss, force_field

    # A new animatin with walkers moved to the new target
    history = point_cloud_advection(pt_cloud, force_field, dt=1, nb_iter=20)
    vis.animate_walk_history(history, log_concentration, output=simulation_path.joinpath("differentiated_walkers.gif"),
                             target=target,
                             force_field=force_field)

    vis.plot_loss(loss, output=simulation_path.joinpath("loss_evolution.png"))


def main(simulation_path: str | Path):
    import snake_ai.physim.visualization as vis

    loader = SimulationLoader(simulation_path)
    simu = loader.load()

    concentration = simu.field
    log_concentration = maths.compute_log_concentration(concentration, epsilon=1e-6)
    force_field = flow.field.spatial_gradient(log_concentration, type=flow.CenteredGrid)
    force_field = maths.clip_gradient_norm(force_field, threashold=1)
    # Plot the concentration field and the walkers inital state
    pt_cloud = simu.point_cloud
    fig, _, _ = vis.plot_walkers_with_concentration(
        pt_cloud, log_concentration, force_field=force_field
    )
    fig.savefig(simulation_path.joinpath("initial_configuration.png"))
    # # Plot a first initial walk to see if it converges to the goal
    history = deterministic_walk_simulation(pt_cloud, force_field, dt=1, nb_iter=100)
    vis.animate_walk_history(
        history,
        log_concentration,
        output=simulation_path.joinpath("initial_walkers.gif"),
        force_field=force_field,
    )

    # Define a target reaching problem and solve it with gradient descent
    # target_center = simu.env.target.center
    # target = flow.vec(x = 15.5, y=5.5)
    # loss = []
    # for e in range(MAX_EPOCH):
    #     value, grad = walk_simulation(pt_cloud, force_field, target, dt=1, nb_iter=20)
    #     print(f"iter : {e}, loss : {value}")
    #     loss.append(value.numpy())
    #     force_field -= grad
    #     force_field = clip_gradient_norm(force_field, threashold=1)

    # # A new animatin with walkers moved to the new target
    # history = point_cloud_advection(pt_cloud, force_field, dt=1, nb_iter=20)
    # vis.animate_walk_history(history, log_concentration, output=simulation_path.joinpath("differentiated_walkers.gif"),
    #                          target=target,
    #                          force_field=force_field)

    # vis.plot_loss(loss, output=simulation_path.joinpath("loss_evolution.png"))


def stochastic_walk(
    point_cloud: flow.Tensor,
    gradient_field: flow.CenteredGrid,
    time_step: int,
    max_iter: int,
    save_step: int,
    D_init: float = 0.1,
):
    i = 0
    history = [point_cloud]
    # D_list = np.linspace(D_init, 0, max_iter)
    while i < max_iter:
        point_cloud += time_step * flow.field.sample(
            gradient_field, point_cloud
        ) + np.sqrt(2 * time_step * D_init) * flow.math.random_normal(point_cloud.shape)
        if i % save_step == 0:
            history.append(point_cloud)
        i += 1
    return history

if __name__ == "__main__":
    dirpath = Path("/home/rcremese/projects/snake-ai/simulations")
    # simulation_path = dirpath.joinpath("GridWorld(20,20)_meta_Tmax=400.0_D=1/seed_0")
    # simulation_path = dirpath.joinpath("RandomObstacles(20,20)_meta_Tmax=400.0_D=1/seed_0")
    # simulation_path = dirpath.joinpath("RandomObstacles(20,20)_pixel_Tmax=400.0_D=1/seed_0")
    simulation_path = dirpath.joinpath("RoomEscape(20,20)_meta_Tmax=400.0_D=1/seed_0")
    # simulation_path = dirpath.joinpath("Slot(20,20)_pixel_Tmax=400.0_D=1/seed_0")
    main(simulation_path)
