from phi.jax import flow

TMAX = 50

@flow.math.jit_compile
def advection_simulation(initial_pos : flow.Tensor, velocity : flow.CenteredGrid, dt : float):
    points = flow.math.to_float(flow.math.copy(initial_pos))
    path_lenght = flow.math.zeros_like(flow.field.l2_loss(points, reduce='vector'))
    for t in range(TMAX):
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

