from phi.jax import flow
import numpy as np

def compute_log_concentration(
    concentration_field: flow.CenteredGrid, epsilon: float = 1e-6
) -> flow.CenteredGrid:
    assert isinstance(concentration_field, flow.CenteredGrid)
    threashold = flow.math.where(
        concentration_field.values > epsilon, concentration_field.values, epsilon
    )
    return flow.CenteredGrid(
        flow.math.log(threashold),
        extrapolation=np.log(epsilon),
        bounds=concentration_field.bounds,
        resolution=concentration_field.resolution,
    )


def clip_gradient_norm(
    force_field: flow.CenteredGrid, threashold=1
) -> flow.CenteredGrid:
    assert isinstance(force_field, flow.CenteredGrid)
    assert (
        "vector" in force_field.values.shape.names
    ), "The force field must contain a 'vector' dimension that contains the values of the force field"
    assert threashold > 0, "The max_bound must be positive"
    norm = flow.math.l2_loss(force_field.values, reduce="vector")
    cliped_values = flow.math.where(
        norm > threashold, force_field.values / norm * threashold, force_field.values
    )
    return flow.CenteredGrid(
        cliped_values, bounds=force_field.bounds, resolution=force_field.resolution
    )


def total_variation(point_cloud: flow.PointCloud) -> flow.Tensor:
    """Compute the total variation of a point cloud containing the trajectories of a set of particles

    Args:
        point_cloud (flow.PointCloud): Point cloud the represent positions of agents over time

    Returns:
        flow.Tensor: total variation of each trajectory
    """
    assert isinstance(point_cloud, flow.PointCloud)
    assert (
        "time" in point_cloud.points.shape.names
    ), "The point cloud must contain a 'time' dimension"
    assert (
        "vector" in point_cloud.points.shape.names
    ), "The point cloud must contain a 'vector' dimension that contains the position of one particle"
    # Compute the position difference between each time step \sigma_{t+1} - \sigma_{t}
    diff = point_cloud.points.time[1:] - point_cloud.points.time[:-1]
    # Compute the total variation of each trajectory
    return flow.math.sum(flow.math.l2_loss(diff, reduce="vector"), dim="time")


def normalized_l2_distance(
    point_cloud: flow.PointCloud, target: flow.Tensor
) -> flow.Tensor:
    """Compute the normalized l2 distance between a point cloud representing trajectories and a target point

    Args:
        point_cloud (flow.PointCloud): Point cloud the represent positions of agents over time
        target (flow.Tensor): position of the target

    Returns:
        flow.Tensor: normalised l2 distance between final position and target position for all trajectories
        \frac{\| \sigma_T - target \|_2}{\| \sigma_0 - target \|_2}
    """
    assert isinstance(point_cloud, flow.PointCloud)
    assert (
        "time" in point_cloud.points.shape.names
    ), "The point cloud must contain a 'time' dimension"
    assert (
        "vector" in point_cloud.points.shape.names
    ), "The point cloud must contain a 'vector' dimension that contains the position of one particle"
    assert (
        "vector" in target.shape.names
    ), "The target must contain a 'vector' dimension that correspond to the position of the target"
    # Compute the total variation of each trajectory
    return flow.math.l2_loss(
        point_cloud.points.time[-1] - target, reduce="vector"
    )  # / flow.math.l2_loss(point_cloud.points.time[0] - target, reduce='vector')
