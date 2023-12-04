import numpy as np
from monai.transforms import (
    LoadImage,
    CropForeground,
    Resize,
    LabelToContour,
    distance_transform_edt,
    Compose,
    KeepLargestConnectedComponent,
)
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from matplotlib import animation
import plotly.graph_objects as go
import scipy.spatial as sp
import skimage as ski  # Library for marching cubes

from snake_ai.diffsim.diffusion import get_laplacian_matrix_from_obstacle_binary_map
import snake_ai.utils.visualization as vis
import scipy.sparse.linalg as spl
from monai.data.meta_tensor import MetaTensor
import logging


def solve_diffusion_equation(volume, source_position) -> np.ndarray:
    assert isinstance(volume, np.ndarray)
    assert isinstance(source_position, np.ndarray)
    init_shape = source_position.shape

    laplace = get_laplacian_matrix_from_obstacle_binary_map(volume)
    solution, callback = spl.cg(-laplace, source_position.flatten())
    print(callback)
    solution = solution.reshape(init_shape)
    return solution


# Idée : diviser la plis grand longueur par 2 permet d'obtenir une segmentation potable
def animate_volume(data: MetaTensor, dim=-1):
    fig, ax = plt.subplots(1, 3)
    frames = data.shape[dim]
    data.swapaxes_(dim, 0)
    contours = LabelToContour()(data)
    distances = distance_transform_edt(data)
    im1 = ax[0].imshow(data[0], cmap="gray", vmin=0, vmax=1)
    im2 = ax[1].imshow(contours[0], cmap="gray", vmin=0, vmax=1)
    im3 = ax[2].imshow(
        distances[0], cmap="gray", vmin=distances.min(), vmax=distances.max()
    )

    def update(frame):
        im1.set_data(data[frame])
        im2.set_data(contours[frame])
        im3.set_data(distances[frame])
        ax[0].set_title(f"Frame {frame}")
        return ax

    anim = animation.FuncAnimation(fig, update, frames=range(frames), interval=100)
    return anim


import plotly.graph_objects as go
import numpy as np


def create_repulsive_field(binary_seg: MetaTensor, max_dist: float):
    # contours = LabelToContour()(binary_seg)
    distances = distance_transform_edt(binary_seg)
    repulsive_field = np.zeros_like(distances)
    indices = np.argwhere((0 < distances) & (distances < max_dist))
    # indices = np.argwhere((distances < max_dist))
    for x, y, z in indices:
        repulsive_field[x, y, z] = (
            0.5
            * ((max_dist - distances[x, y, z]) / (max_dist * distances[x, y, z])) ** 2
        )
    return repulsive_field


def create_attractive_field(binary_seg: MetaTensor, position: np.ndarray):
    assert len(binary_seg.shape) == 3
    assert position.shape == (3,)
    shape = binary_seg.shape
    meshgrid = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    attractive_field = np.linalg.norm(meshgrid - position.reshape(-1, 1, 1, 1), axis=0)
    return np.where(binary_seg > 0, 0.5 * attractive_field**2, 0)


def plot_3d_volume(values: np.ndarray):
    x = np.arange(0, values.shape[0])
    y = np.arange(0, values.shape[1])
    z = np.arange(0, values.shape[2])

    X, Y, Z = np.meshgrid(x, y, z)
    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values.flatten(),
            # isomin=-10,
            # isomax=10,
            opacity=0.3,  # needs to be small to see through all surfaces
            surface_count=21,  # needs to be a large number for good volume rendering
        )
    )
    fig.show()


def generate_interactive_3D_surface(data: MetaTensor):
    """Create an empty scatter3d trace for the interactive points"""
    fig = plot_3d_surface(data)
    scatter_trace = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode="markers",
        marker=dict(
            size=8,
            color="green",
            opacity=1,
        ),
        name="Interactive Points",
    )

    def click_callback(trace, points, state):
        # Get the coordinates of the clicked point
        x, y, z = points.point_inds[0], points.point_inds[1], points.point_inds[2]

        # Add the clicked point to the scatter trace
        scatter_trace.x = scatter_trace.x + [x]
        scatter_trace.y = scatter_trace.y + [y]
        scatter_trace.z = scatter_trace.z + [z]
        print(scatter_trace.x, scatter_trace.y, scatter_trace.z)
        # Update the figure with the new scatter trace
        fig.add_trace(scatter_trace)

    fig.data[0].on_click(click_callback)
    return fig


def plot_3d_surface(data: MetaTensor, alpha=1.0):
    vertices, triangles, _, _ = ski.measure.marching_cubes(data.numpy(), 0.5)
    fig = ff.create_trisurf(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        simplices=triangles,
        title="Lung representation",
    )
    return fig


def main():
    div_factor = 2
    k_rep, k_atr = 1e2, 1e-3
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    data = LoadImage(ensure_channel_first=True)("datas/ATM_001_0000.nii.gz")
    croper = CropForeground()
    croped_data = croper(data)
    print(croped_data.shape)
    # anim = animate_volume(croped_data.squeeze())
    # plt.show()
    new_dim = int(np.max(croped_data.shape) // div_factor)
    composer = Compose(
        [
            Resize(new_dim, size_mode="longest", mode="nearest"),
            KeepLargestConnectedComponent(applied_labels=[1], connectivity=1),
            # LabelToContour(),
        ]
    )
    resized_data = composer(croped_data)
    ## Figure to find the center of the lung
    # fig, ax = plt.subplots()
    # resized_data = resized_data.squeeze()
    # ax.imshow(resized_data[:, :, -1].T, cmap="gray")
    # ax.axvline(x=resized_data.shape[0] // 2, color="red")
    # ax.axhline(y=resized_data.shape[1] // 2, color="red")
    # ax.set(xlabel="x", ylabel="y", title=f"z={resized_data.shape[2]}")
    # plt.show()
    # resized_data = Resize((100, 100, 100), size_mode="all", mode="nearest")(croped_data)
    logging.debug(f"New shape : {resized_data.shape}")
    logging.info("Plotting volumes")
    anim1 = animate_volume(resized_data.squeeze())
    repulsive_field = create_repulsive_field(resized_data.squeeze(), max_dist=5.0)
    attractive_field = create_attractive_field(
        resized_data.squeeze(), np.array([65, 27, 230])
    )

    anim2 = vis.animate_volume(
        k_rep * repulsive_field + k_atr * attractive_field, axis=2
    )
    plt.show()

    ## Create a potential field and a simulation
    from snake_ai.diffsim.field import ScalarField
    from snake_ai.diffsim.walk_simulation import WalkerSimulationStoch3D
    from snake_ai.envs.geometry import Cube
    import taichi as ti

    ti.init(arch=ti.gpu)
    logging.info("Creating potential field")
    potential_field = ScalarField(
        -(k_rep * repulsive_field + k_atr * attractive_field),
        bounds=Cube(0, 0, 0, *repulsive_field.shape),
    )
    init_pos = np.array([[40, 40, 85], [80, 45, 85]])
    simu = WalkerSimulationStoch3D(
        init_pos, potential_field, t_max=100, dt=1e-1, diffusivity=0
    )
    simu.optimize(target_pos=np.array([65, 27, 230]), lr=1.0e-1, max_iter=200)

    ## Plot the result of the simulation
    logging.info("Plotting simulation")
    fig = plot_3d_surface(resized_data.squeeze(), alpha=0.5)
    trajectories = simu.positions
    for i in range(trajectories.shape[0]):
        fig.add_trace(
            go.Scatter3d(
                x=trajectories[i, :, 0],
                y=trajectories[i, :, 1],
                z=trajectories[i, :, 2],
                mode="lines",
                name=f"walker {i}",
                # colorscale="Viridis",
                # color=color,
            )
        )
    fig.show()

    # fig = plot_3d_surface(resized_data.squeeze())

    # Z max, x and y midle
    # obstacle_map = LabelToContour()(resized_data).squeeze().numpy()
    obstacle_map = ~resized_data.squeeze().numpy().astype(int)
    # anim2 = vis.animate_volume(obstacle_map)
    # plt.show()
    source_position = np.zeros_like(obstacle_map)
    source_position[130 // div_factor, 54 // div_factor, -1] = 100
    logging.info("Solving diffusion equation")
    solution = solve_diffusion_equation(obstacle_map, source_position)
    np.save("solution.npy", solution)

    # solution = np.load("solution.npy")
    log_sol = np.log(np.where(solution > 1e-10, solution, 1e-10))
    logging.info("Plotting solution")
    anim = vis.animate_volume(log_sol)
    plot_3d_volume(log_sol)
    plt.show()
    # anim = animate_volume(resized_data.squeeze())
    # plt.show()


if __name__ == "__main__":
    main()
