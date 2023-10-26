import nibabel as nib
import numpy as np
from monai.transforms import LoadImage, CropForeground, Resize, LabelToContour
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from matplotlib import animation
import plotly.graph_objects as go
import scipy.spatial as sp
import torch
import mcubes  # Library for marching cubes
from snake_ai.taichi.diffusion import get_laplacian_matrix_from_obstacle_binary_map
import scipy.sparse.linalg as spl
import scipy.sparse as sps


def solve_diffusion_equation(volume, source_position) -> np.ndarray:
    assert isinstance(volume, np.ndarray)
    assert isinstance(source_position, np.ndarray)
    init_shape = source_position.shape

    laplace = get_laplacian_matrix_from_obstacle_binary_map(volume)
    solution = spl.spsolve(-laplace, source_position.flatten())
    solution = solution.reshape(init_shape)
    np.save("solution.npy", solution)
    return solution


# Idée : diviser la plis grand longueur par 2 permet d'obtenir une segmentation potable
def animate_volume(data: torch.Tensor, dim=-1):
    fig, ax = plt.subplots(1, 2)
    frames = data.shape[dim]
    data.swapaxes_(dim, 0)
    contours = LabelToContour()(data)
    im1 = ax[0].imshow(data[0], cmap="gray")
    im2 = ax[1].imshow(contours[0], cmap="gray")

    def update(frame):
        im1.set_data(data[frame])
        im2.set_data(contours[frame])
        ax[0].set_title(f"Frame {frame}")
        return ax

    anim = animation.FuncAnimation(fig, update, frames=range(frames), interval=100)
    return anim


import plotly.graph_objects as go
import numpy as np

X, Y, Z = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
values = np.sin(np.pi * X) * np.cos(np.pi * Z) * np.sin(np.pi * Y)


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


def plot_3d_surface(data: torch.Tensor, point=None):
    vertices, triangles = mcubes.marching_cubes(data.numpy(), 0.5)
    fig = ff.create_trisurf(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        simplices=triangles,
        title="Lung representation",
    )
    if point is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[point[0]],
                y=[point[1]],
                z=[point[2]],
                mode="markers",
                marker=dict(size=5, color="green"),
            )
        )
    fig.show()


def main():
    data = LoadImage(ensure_channel_first=True)("datas/ATM_001_0000.nii.gz")
    croper = CropForeground()
    croped_data = croper(data)
    print(croped_data.shape)
    anim = animate_volume(croped_data.squeeze())
    plt.show()
    dim = np.argmax(croped_data.shape[1:])
    new_shape = list(croped_data.shape[1:])
    new_shape[dim] = new_shape[dim] // 10
    # resized_data = Resize(new_shape, size_mode="all", mode="nearest")(croped_data)
    resized_data = Resize(new_shape[dim], size_mode="longest", mode="nearest")(
        croped_data
    )
    # resized_data = Resize((100, 100, 100), size_mode="all", mode="nearest")(croped_data)
    print(resized_data.shape)
    spacial_shape = resized_data.shape[1:]
    plot_3d_surface(resized_data.squeeze())
    # Z max, x and y midle
    obstacle_map = LabelToContour()(resized_data).squeeze().numpy()
    source_position = np.zeros_like(obstacle_map)
    source_position[
        spacial_shape[0] // 2, spacial_shape[1] // 2, spacial_shape[2] - 1
    ] = 1

    point = np.array(
        [
            int(spacial_shape[0] * 3 / 5),
            int(spacial_shape[1] * 0.5),
            spacial_shape[2] - 2,
        ]
    )
    solution = solve_diffusion_equation(obstacle_map, source_position)
    plot_3d_volume(solution)
    # anim = animate_volume(resized_data.squeeze())
    # plt.show()


if __name__ == "__main__":
    main()
