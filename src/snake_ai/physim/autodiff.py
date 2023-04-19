from snake_ai.envs import RoomEscape, SlotEnv
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
def advection_simulation(initial_pos : flow.Tensor, velocity : flow.CenteredGrid, dt : float):
    points = flow.math.to_float(flow.math.copy(initial_pos))
    path_lenght = flow.math.zeros_like(flow.field.l2_loss(points, reduce='vector'))
    for t in range(MAX_ITER):
        update = dt * flow.field.sample(velocity, points)
        points += update
        path_lenght += flow.field.l2_loss(update, reduce='vector')
    return points, path_lenght

def position_loss(final_pos : flow.Tensor, initial_pos : flow.Tensor):
    return flow.math.mean(flow.field.l2_loss(final_pos, reduce='vector') / flow.field.l2_loss(initial_pos, reduce='vector'))

# @flow.math.functional_gradient
def optimization_step(initial_pos : flow.Tensor, velocity : flow.CenteredGrid, dt : float, gamma : float):
    final_pos, path_length = advection_simulation(initial_pos, velocity, dt)
    return position_loss(final_pos, initial_pos) + gamma * flow.math.mean(path_length)

def deterministic_walk(point_cloud : flow.Tensor, gradient_field : flow.CenteredGrid, time_step : int, goal : flow.Tensor, max_iter : int, save_step : int):
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
        simu = DiffusionSimulation(env, t_max=TMAX, dt=DT, init_value=INIT_VALUE, history=True, stationary=True)
        simu.reset()
        simu.start()
        SimulationWritter(dir_path).write(simu)

    # animate_simulation_history(simu, dir_path.joinpath(f"concentration_diffusion.gif"))
    # Get log-concentration field
    threashold = flow.math.where(simu.field.values > EPS, simu.field.values, EPS)
    log_concentration = flow.CenteredGrid(flow.math.log(threashold), extrapolation=np.log(EPS), bounds=simu.field.bounds, resolution=simu.field.resolution)
    log_grad = flow.field.spatial_gradient(log_concentration)
    ## Visualisation
    plot_concentration_with_gradient(simu.field, dir_path.joinpath(f"concentration.png"))
    plot_concentration_with_gradient(log_concentration, dir_path.joinpath(f"concentration_log.png"))
    # history = deterministic_walk(simu.point_cloud, log_grad, time_step=10, max_iter=100, save_step=10)
    # animate_walk_history(log_concentration, history, dir_path.joinpath(f"walk_init.gif"))

    ## Optimization steps
    lr = 0.1
    grad_eval = flow.math.functional_gradient(optimization_step, wrt='velocity')
    for step in range(200):
        loss, grad = grad_eval(simu.point_cloud, log_grad, dt=10, gamma=0.1)
        log_grad -= lr * grad
        print(loss)

    history = deterministic_walk(simu.point_cloud, log_grad, time_step=10, max_iter=100, save_step=10)
    animate_walk_history(log_concentration, history, dir_path.joinpath(f"walk_final.gif"))

    # plt.show()

if __name__ == '__main__':
    main()