##
# @author  <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2023-06-24 10:07:34 am
 # @copyright MIT License
 #
from snake_ai.envs import RoomEscape, SlotEnv, RandomObstaclesEnv
from snake_ai.physim.simulation import DiffusionSimulation
from snake_ai.physim.solver import DiffusionSolver
from snake_ai.physim.visualization import plot_concentration_with_gradient, animate_simulation_history, animate_walk_history
from snake_ai.utils.io import SimulationWritter, SimulationLoader
from phi.jax import flow
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

MAX_ITER = 100

# @partial(flow.math.jit_compile, auxiliary_args='dt,nb_iter')
def point_cloud_advection(point_cloud : flow.PointCloud, force_field : flow.CenteredGrid, dt : float, nb_iter : int = MAX_ITER):
    nb_iter = int(nb_iter)
    history = nb_iter * [0]
    history[0] = point_cloud
    for i in range(1, nb_iter):
        point_cloud = flow.advect.advect(point_cloud, force_field, dt=dt)
        history[i] = point_cloud
    return flow.field.stack(history, flow.batch('time'))

def compute_log_concentration(concentration_field : flow.CenteredGrid, epsilon : float = 1e-6) -> flow.CenteredGrid:
    assert isinstance(concentration_field, flow.CenteredGrid)
    threashold = flow.math.where(concentration_field.values > epsilon, concentration_field.values, epsilon)
    return flow.CenteredGrid(flow.math.log(threashold), extrapolation=np.log(epsilon), bounds=concentration_field.bounds, resolution=concentration_field.resolution)

def clip_gradient_norm(force_field : flow.CenteredGrid, threashold = 1) -> flow.CenteredGrid:
    assert isinstance(force_field, flow.CenteredGrid)
    assert 'vector' in force_field.values.shape.names, "The force field must contain a 'vector' dimension that contains the values of the force field"
    assert threashold > 0, "The max_bound must be positive"
    norm = flow.math.l2_loss(force_field.values, reduce='vector')
    cliped_values = flow.math.where(norm > threashold, force_field.values / norm * threashold, force_field.values)
    return flow.CenteredGrid(cliped_values, bounds=force_field.bounds, resolution=force_field.resolution)

def total_variation(point_cloud : flow.PointCloud) -> flow.Tensor:
    """Compute the total variation of a point cloud containing the trajectories of a set of particles

    Args:
        point_cloud (flow.PointCloud): Point cloud the represent positions of agents over time

    Returns:
        flow.Tensor: total variation of each trajectory
    """
    assert isinstance(point_cloud, flow.PointCloud)
    assert 'time' in point_cloud.points.shape.names, "The point cloud must contain a 'time' dimension"
    assert 'vector' in point_cloud.points.shape.names, "The point cloud must contain a 'vector' dimension that contains the position of one particle"
    # Compute the position difference between each time step \sigma_{t+1} - \sigma_{t}
    diff = point_cloud.points.time[1:] - point_cloud.points.time[:-1]
    # Compute the total variation of each trajectory
    return flow.math.sum(flow.math.l2_loss(diff, reduce='vector'), dim='time')

def normalized_l2_distance(point_cloud : flow.PointCloud, target : flow.Tensor) -> flow.Tensor:
    """Compute the normalized l2 distance between a point cloud representing trajectories and a target point

    Args:
        point_cloud (flow.PointCloud): Point cloud the represent positions of agents over time
        target (flow.Tensor): position of the target

    Returns:
        flow.Tensor: normalised l2 distance between final position and target position for all trajectories
        \frac{\| \sigma_T - target \|_2}{\| \sigma_0 - target \|_2}
    """
    assert isinstance(point_cloud, flow.PointCloud)
    assert 'time' in point_cloud.points.shape.names, "The point cloud must contain a 'time' dimension"
    assert 'vector' in point_cloud.points.shape.names, "The point cloud must contain a 'vector' dimension that contains the position of one particle"
    assert 'vector' in target.shape.names, "The target must contain a 'vector' dimension that correspond to the position of the target"
    # Compute the total variation of each trajectory
    return flow.math.l2_loss(point_cloud.points.time[-1] - target, reduce='vector') #/ flow.math.l2_loss(point_cloud.points.time[0] - target, reduce='vector')

@partial(flow.functional_gradient, wrt='force_field')
def walk_simulation(point_cloud : flow.PointCloud, force_field : flow.CenteredGrid, target : flow.Tensor, dt : float, nb_iter : int = MAX_ITER):
    advected_sim = flow.jit_compile(point_cloud_advection, auxiliary_args='dt,nb_iter')
    trajectories = advected_sim(point_cloud, force_field, dt=dt, nb_iter=nb_iter)
    return flow.math.mean(normalized_l2_distance(trajectories, target))

def main(simulation_path : str | Path):
    import snake_ai.physim.visualization as vis
    loader = SimulationLoader(simulation_path)
    simu = loader.load()

    concentration = simu.field
    log_concentration = compute_log_concentration(concentration, epsilon=1e-6)
    force_field = flow.field.spatial_gradient(log_concentration, type=flow.CenteredGrid)
    force_field = clip_gradient_norm(force_field, threashold=1)
    # Plot the concentration field and the walkers inital state
    pt_cloud = simu.point_cloud
    # fig, _, _ = vis.plot_walkers_with_concentration(log_concentration, pt_cloud, force_field=force_field)
    # fig.savefig(simulation_path.joinpath("initial_configuration.png"))
    # Plot a first initial walk to see if it converges to the goal
    # history = point_cloud_advection(pt_cloud, force_field, dt=1, nb_iter=100)
    # vis.animate_walk_history(log_concentration, history, output=simulation_path.joinpath("initial_walkers.gif"), force_field=force_field)

    # Define a target reaching problem and solve it with gradient descent
    target = flow.vec(x = 5, y=5)
    loss = []
    for e in range(100):
        value, grad = walk_simulation(pt_cloud, force_field, target, dt=1, nb_iter=20)
        print(f"iter : {e}, loss : {value}")
        loss.append(value.numpy())
        force_field -= grad
        force_field = clip_gradient_norm(force_field, threashold=1)

    # A new animatin with walkers moved to the new target
    history = point_cloud_advection(pt_cloud, force_field, dt=1, nb_iter=100)
    vis.animate_walk_history(log_concentration, history, output=simulation_path.joinpath("differentiated_walkers.gif"),
                             target=target,
                             force_field=force_field)

    vis.plot_loss(loss, output=simulation_path.joinpath("loss_evolution.png"))
    # flow.vis.plot([force_field, grad, flow.math.l2_loss(grad, reduce='vector')])
    # plt.show()


@flow.math.jit_compile
def advection_simulation(initial_pos : flow.Tensor, velocity : flow.CenteredGrid, dt : float):
    points = flow.math.to_float(flow.math.copy(initial_pos))
    path_lenght = flow.math.zeros_like(flow.field.l2_loss(points, reduce='vector'))
    for t in range(MAX_ITER):
        update = dt * flow.field.sample(velocity, points)
        points += update
        path_lenght += flow.field.l2_loss(update, reduce='vector')
    return points, path_lenght

def position_loss(final_pos : flow.Tensor, initial_pos : flow.Tensor, target_pos : flow.Tensor):
    return flow.math.mean(flow.field.l2_loss(final_pos - target_pos, reduce='vector') / flow.field.l2_loss(initial_pos, reduce='vector'))

# @flow.math.functional_gradient
def optimization_step(initial_pos : flow.Tensor, target_pos : flow.Tensor, velocity : flow.CenteredGrid, dt : float, gamma : float):
    final_pos, path_length = advection_simulation(initial_pos, velocity, dt)
    return position_loss(final_pos, initial_pos, target_pos) + gamma * flow.math.mean(path_length)

def deterministic_walk(point_cloud : flow.Tensor, gradient_field : flow.CenteredGrid, time_step : int, max_iter : int, save_step : int):
    i = 0
    history = [point_cloud]
    # jited_advection = flow.math.jit_compile(flow.advect.advect)
    while i < max_iter:
        point_cloud += time_step * flow.field.sample(gradient_field, point_cloud)

        # point_cloud : flow.PointCloud = flow.advect.advect(point_cloud, gradient_field, dt=time_step)
        # point_cloud : flow.PointCloud = jited_advection(point_cloud, gradient_field, time_step)
        if i % save_step == 0:
            history.append(point_cloud)
        i +=1
    return history

def stochastic_walk(point_cloud : flow.Tensor, gradient_field : flow.CenteredGrid, time_step : int, max_iter : int, save_step : int, D_init : float = 0.1):
    i = 0
    history = [point_cloud]
    # D_list = np.linspace(D_init, 0, max_iter)
    while i < max_iter:
        point_cloud += time_step * flow.field.sample(gradient_field, point_cloud) + np.sqrt(2 * time_step * D_init) * flow.math.random_normal(point_cloud.shape)
        if i % save_step == 0:
            history.append(point_cloud)
        i +=1
    return history

def trajectory_optimization(simu : DiffusionSimulation, target : flow.Tensor, log_grad : flow.CenteredGrid):
    lr = 0.01
    target = flow.tensor(flow.math.vec(x = simu.env.goal.x + simu.env.goal.width / 2, y=simu.env.goal.y + simu.env.goal.height / 2))
    print(target)
    grad_eval = flow.math.functional_gradient(optimization_step, wrt='velocity')
    loss_evol = []
    for step in range(150):
        loss, grad = grad_eval(simu.point_cloud, target, log_grad, dt=10, gamma=0.1)
        log_grad -= lr * grad
        print(loss)
        loss_evol.append(loss.numpy())
    return loss_evol, log_grad

# def main():
#     INIT_VALUE = 1e6
#     TMAX = 1000
#     DIFFUSIVITY = 1
#     SEED = 0
#     EPS = 1e-6
#     DT = 1


#     save_path = Path(__file__).parents[3].joinpath("datas").resolve(strict=True)
#     simu_name = f"slotEnv_diffusion_Tmax={TMAX}_D={DIFFUSIVITY}_seed={SEED}"
#     dir_path = save_path.joinpath(simu_name)
#     if dir_path.exists():
#         simu = SimulationLoader(dir_path).load()
#     else:
#         env = SlotEnv(seed=SEED)
#         print("Starting simulation...")
#         simu = DiffusionSimulation(env, t_max=TMAX, dt=DT, init_value=INIT_VALUE, history=True, stationary=True)
#         simu.reset()
#         simu.start()
#         SimulationWritter(dir_path).write(simu)

#     print("Diffusion simulation...")
#     # animate_simulation_history(simu.history, dir_path.joinpath(f"concentration_evolution.gif"))
#     # Get log-concentration field
#     print("Computing log-concentration field...")
#     threashold = flow.math.where(simu.field.values > EPS, simu.field.values, EPS)
#     log_concentration = flow.CenteredGrid(flow.math.log(threashold), extrapolation=np.log(EPS), bounds=simu.field.bounds, resolution=simu.field.resolution)
#     log_grad = flow.field.spatial_gradient(log_concentration)
#     ## Visualisation
#     # plot_concentration_with_gradient(simu.field, dir_path.joinpath(f"concentration.png"))
#     # plot_concentration_with_gradient(log_concentration, dir_path.joinpath(f"concentration_log.png"))
#     # print("First walk...")
#     # agent = flow.tensor(flow.math.vec(x = simu.env.agent.position.centerx, y=simu.env.agent.position.centery))
#     history = deterministic_walk(simu.point_cloud, log_grad, time_step=10, max_iter=300, save_step=10)
#     animate_walk_history(log_concentration, history, dir_path.joinpath(f"walk.gif"))
#     # stochastic_history = stochastic_walk(flow.tensor(5 * [agent, ], flow.instance('point')), log_grad, time_step=10, max_iter=500, save_step=10, D_init=0.01)
#     # animate_walk_history(log_concentration, stochastic_history, dir_path.joinpath(f"stochastic_walk.gif"))

#     ## Optimization steps
#     # print("Starting optimization...")
#     # loss_evol, new_log = trajectory_optimization(simu, simu.env.goal, log_grad)

#     # print("Saving results...")
#     # history = deterministic_walk(simu.point_cloud, new_log, time_step=10, max_iter=150, save_step=10)
#     # animate_walk_history(log_concentration, history, dir_path.joinpath(f"walk_final.gif"))
#     # fig, ax = plt.subplots(dpi=300)
#     # ax.plot(loss_evol)
#     # ax.set(xlabel="Iteration", ylabel="Loss")
#     # fig.savefig(dir_path.joinpath("loss_evolution.png"))


if __name__ == '__main__':
    dirpath = Path("/home/rocremes/projects/snake-ai/simulations")
    simulation_path = dirpath.joinpath("GridWorld(20,20)_meta_Tmax=400.0_D=1/seed_0")
    main(simulation_path)