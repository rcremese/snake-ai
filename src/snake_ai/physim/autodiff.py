from snake_ai.envs import RoomEscape, SlotEnv, RandomObstaclesEnv
from snake_ai.physim.simulation import DiffusionSimulation
from snake_ai.physim.solver import DiffusionSolver
from snake_ai.physim.visualization import plot_concentration_with_gradient, animate_simulation_history, animate_walk_history
from snake_ai.utils.io import SimulationWritter, SimulationLoader
from phi.jax import flow
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

MAX_ITER = 100

@flow.math.jit_compile
def point_cloud_advection(point_cloud : flow.PointCloud, force_field : flow.CenteredGrid, dt : float, t_max : float):
    # history = [point_cloud]
    t = 0
    while t < t_max:
        point_cloud = flow.advect(point_cloud, force_field, dt=dt, t_max=t_max)
        # history.append(point_cloud)
        t += dt
    return point_cloud

def convert_concentration_field(concentration_field : flow.CenteredGrid, threshold : float = 1e-6):
    threashold = flow.math.where(concentration_field.values > threshold, concentration_field.values, threshold)
    return flow.CenteredGrid(flow.math.log(threashold), extrapolation=np.log(threshold), bounds=concentration_field.bounds, resolution=concentration_field.resolution)

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

def main():
    INIT_VALUE = 1e6
    TMAX = 1000
    DIFFUSIVITY = 1
    SEED = 0
    EPS = 1e-6
    DT = 1


    save_path = Path(__file__).parents[3].joinpath("datas").resolve(strict=True)
    simu_name = f"slotEnv_diffusion_Tmax={TMAX}_D={DIFFUSIVITY}_seed={SEED}"
    dir_path = save_path.joinpath(simu_name)
    if dir_path.exists():
        simu = SimulationLoader(dir_path).load()
    else:
        env = SlotEnv(seed=SEED)
        print("Starting simulation...")
        simu = DiffusionSimulation(env, t_max=TMAX, dt=DT, init_value=INIT_VALUE, history=True, stationary=True)
        simu.reset()
        simu.start()
        SimulationWritter(dir_path).write(simu)

    print("Diffusion simulation...")
    # animate_simulation_history(simu.history, dir_path.joinpath(f"concentration_evolution.gif"))
    # Get log-concentration field
    print("Computing log-concentration field...")
    threashold = flow.math.where(simu.field.values > EPS, simu.field.values, EPS)
    log_concentration = flow.CenteredGrid(flow.math.log(threashold), extrapolation=np.log(EPS), bounds=simu.field.bounds, resolution=simu.field.resolution)
    log_grad = flow.field.spatial_gradient(log_concentration)
    ## Visualisation
    # plot_concentration_with_gradient(simu.field, dir_path.joinpath(f"concentration.png"))
    # plot_concentration_with_gradient(log_concentration, dir_path.joinpath(f"concentration_log.png"))
    # print("First walk...")
    # agent = flow.tensor(flow.math.vec(x = simu.env.agent.position.centerx, y=simu.env.agent.position.centery))
    history = deterministic_walk(simu.point_cloud, log_grad, time_step=10, max_iter=300, save_step=10)
    animate_walk_history(log_concentration, history, dir_path.joinpath(f"walk.gif"))
    # stochastic_history = stochastic_walk(flow.tensor(5 * [agent, ], flow.instance('point')), log_grad, time_step=10, max_iter=500, save_step=10, D_init=0.01)
    # animate_walk_history(log_concentration, stochastic_history, dir_path.joinpath(f"stochastic_walk.gif"))

    ## Optimization steps
    # print("Starting optimization...")
    # loss_evol, new_log = trajectory_optimization(simu, simu.env.goal, log_grad)

    # print("Saving results...")
    # history = deterministic_walk(simu.point_cloud, new_log, time_step=10, max_iter=150, save_step=10)
    # animate_walk_history(log_concentration, history, dir_path.joinpath(f"walk_final.gif"))
    # fig, ax = plt.subplots(dpi=300)
    # ax.plot(loss_evol)
    # ax.set(xlabel="Iteration", ylabel="Loss")
    # fig.savefig(dir_path.joinpath("loss_evolution.png"))

if __name__ == '__main__':
    pass