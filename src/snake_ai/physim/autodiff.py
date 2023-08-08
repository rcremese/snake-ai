##
# @author  <robin.cremese@gmail.com>
# @file Description
# @desc Created on 2023-06-24 10:07:34 am
# @copyright MIT License
#
from snake_ai.envs import RoomEscape, SlotEnv, RandomObstaclesEnv
from snake_ai.physim.simulation import Simulation
from snake_ai.physim import maths
from snake_ai.utils.io import SimulationWritter, SimulationLoader
from phi.jax import flow
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from typing import List, Union, Optional

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

# @partial(flow.math.jit_compile, auxiliary_args="dt,nb_iter")
def stochastic_walk_simulation(
    point_cloud: flow.PointCloud,
    force_field: flow.CenteredGrid,
    dt: float,
    nb_iter: int,
    diffusivity: float = 0.01,
):
    i = 1
    history = [point_cloud]
    while i < int(nb_iter):
        points = point_cloud.elements
        update = dt * flow.field.sample(force_field, points) + np.sqrt(2 * dt * diffusivity) * flow.math.random_normal(points.shape) 
        point_cloud = point_cloud.with_elements(points.shifted(update))
        history.append(point_cloud)
        i += 1
    return flow.field.stack(history, flow.batch("time"))

## TODO : integrer ce code dans trajectory optimization
# @partial(flow.functional_gradient, wrt="force_field")
# def simulation(
#     point_cloud: flow.PointCloud,
#     force_field: flow.CenteredGrid,
#     target: flow.Tensor,
#     dt: float,
#     nb_iter: int = MAX_ITER,
# ):
#     trajectories = deterministic_walk_simulation(
#         point_cloud, force_field, dt=dt, nb_iter=nb_iter
#     )
#     return flow.math.mean(maths.normalized_l2_distance(trajectories, target))

# def close_to_target(point_cloud: flow.PointCloud, force_field : flow.CenteredGrid, target: flow.Tensor, threashold: float = 1):
#     target = flow.vec(x = 15.5, y=5.5)
#     maths.clip_gradient_norm(force_field, threashold=1)
#     loss = []
#     for e in range(MAX_EPOCH):
#         diff_simulation = flow.functional_gradient(simulation, wrt="force_field")
#         value, grad = diff_simulation(point_cloud, force_field, target, dt=1, nb_iter=20)
#         print(f"iter : {e}, loss : {value}")
#         loss.append(value.numpy())
#         force_field -= grad
#         force_field = maths.clip_gradient_norm(force_field, threashold=1)
#     return loss, force_field

    # A new animatin with walkers moved to the new target
    history = point_cloud_advection(pt_cloud, force_field, dt=1, nb_iter=20)
    vis.animate_walk_history(history, log_concentration, output=simulation_path.joinpath("differentiated_walkers.gif"),
                             target=target,
                             force_field=force_field)

    vis.plot_loss(loss, output=simulation_path.joinpath("loss_evolution.png"))

def trajectory_optimization(point_cloud : flow.PointCloud, initial_field : flow.CenteredGrid, target : flow.geom.Point, obstacles: List[flow.Geometry], 
                            max_epoch : int, lr : float, nb_iter : int, dt : float, gradient_clip : Optional[float] = 1, diffusivity : float = 0):
    assert gradient_clip is None or gradient_clip > 0, f"Expected gradient_clip to be > 0 or None, get {gradient_clip}"
    if gradient_clip is None:
        force_field = initial_field
    else:
        force_field = maths.clip_gradient_norm(initial_field, threashold=gradient_clip)
        
    # diff_simulation = flow.functional_gradient(simulation, wrt="force_field")
    
    @partial(flow.functional_gradient, wrt="force_field")
    def simulation(
        point_cloud: flow.PointCloud,
        force_field: flow.CenteredGrid,
    ):
        trajectories = stochastic_walk_simulation(point_cloud, force_field, dt=dt, nb_iter=nb_iter, diffusivity=diffusivity)
        return flow.math.mean(maths.normalized_l2_distance(trajectories, target))

    loss = []
    best_loss = np.inf
    best_field = force_field
    obstacles = flow.union(obstacles) 
    
    for epoch in range(max_epoch):
        value, grad = simulation(point_cloud, force_field)
        # value, grad = diff_simulation(point_cloud, force_field, target=target, dt=dt, nb_iter=nb_iter)
        print(f"iter : {epoch}, loss : {value}")
        loss.append(value.numpy())
        force_field -= lr * grad
        # post processing of the force field
        force_field = flow.field.where(obstacles, 0, force_field)
        if gradient_clip is not None:
            force_field = maths.clip_gradient_norm(force_field, threashold=gradient_clip)
        
        if value < best_loss:
            best_loss = value
            best_field = force_field

    return loss, best_field


def main(simulation_path: Union[str, Path]):
    import snake_ai.physim.visualization as vis

    loader = SimulationLoader(simulation_path)
    simu = loader.load()

    concentration = simu.field
    log_concentration = maths.compute_log_concentration(concentration, epsilon=1e-6)
    force_field = flow.field.spatial_gradient(log_concentration, type=flow.CenteredGrid)
    force_field = maths.clip_gradient_norm(force_field, threashold=1)
    masked_values = flow.math.where(simu.obstacles.values, 0, force_field.values)    
    force_field = force_field.with_values(masked_values)
    # Plot the concentration field and the walkers inital state
    pt_cloud = simu.point_cloud
    # fig, _, _ = vis.plot_walkers_with_concentration(
    #     pt_cloud, log_concentration, force_field=force_field
    # )
    # fig.savefig(simulation_path.joinpath("initial_configuration.png"))
    # # Plot a first initial walk to see if it converges to the goal
    history = stochastic_walk_simulation(pt_cloud, force_field, dt=1, nb_iter=10)
    obstacles = simu.obstacles
    print(history * obstacles)
    # vis.animate_walk_history(
    #     history,
    #     log_concentration,
    #     output=simulation_path.joinpath("test.gif"),
    #     force_field=force_field,
    # )

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


if __name__ == "__main__":
    dirpath = Path("/home/rcremese/projects/snake-ai/simulations")
    # simulation_path = dirpath.joinpath("GridWorld(20,20)_meta_Tmax=400.0_D=1/seed_0")
    # simulation_path = dirpath.joinpath("RandomObstacles(20,20)_meta_Tmax=400.0_D=1/seed_0")
    # simulation_path = dirpath.joinpath("RandomObstacles(20,20)_pixel_Tmax=400.0_D=1/seed_0")
    simulation_path = dirpath.joinpath("RoomEscape(20,20)_meta_Tmax=1600.0_D=1/seed_0")
    # simulation_path = dirpath.joinpath("Slot(20,20)_pixel_Tmax=400.0_D=1/seed_0")
    main(simulation_path)
