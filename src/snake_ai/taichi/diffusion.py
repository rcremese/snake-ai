import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.signal as sg


def main():
    row, col = 10, 10

    def coord2index(i, j):
        return i * col + j

    source = np.zeros((row, col), dtype=np.float32)
    source[5, 0] = 100

    index2delet = [(5, 5), (5, 6), (6, 5), (6, 6)]
    indexes = [coord2index(i, j) for i, j in index2delet]

    ones = np.ones(col * row)
    # ones = np.ones(col)
    # 5 stencil laplace matrix
    datas = np.array([ones, ones, -4 * ones, ones, ones])
    offsets = np.array([-col, -1, 0, 1, col])
    # 9 stencil laplace matrix
    # datas = np.array([ones, ones, ones, ones, -8 * ones, ones, ones, ones, ones])
    # offsets = np.array([-n - 1, -n, -n + 1, -1, 0, 1, n - 1, n, n + 1])

    laplace = sp.diags(
        datas, offsets, shape=(col * row, col * row), dtype=np.float32
    ).tolil()
    laplace[0, row] = 0
    laplace[row, 0] = 0
    for idx in range(row, row * col, row):
        laplace[idx - 1, idx] = 0
        laplace[idx, idx - 1] = 0

    ## Base blockconstruction
    # base_block = sp.diags(datas, offsets, shape=(col, col), dtype=np.float32)
    # laplace = sp.kron(sp.eye(row), base_block, format="lil")

    # outer_diag = np.ones(row * col)
    # laplace.setdiag(outer_diag, -col)
    # laplace.setdiag(outer_diag, col)

    for idx in indexes:
        laplace[idx, :] = 0
        laplace[:, idx] = 0
        laplace[idx, idx] = 1

    ## lil_matrix construction
    # laplace = sp.lil_matrix((row * col, row * col), dtype=np.float32)
    # for data, offset in zip(datas, offsets):
    #     laplace.setdiag(data, offset)
    # for k in range(0, row * col, col):
    #     laplace[::col, ::col] = 0

    # laplace.setdiag(datas[0], offsets[0])

    solver = sp.linalg.factorized(-laplace.tocsc())
    solution = solver(source.flatten())
    solution = solution.reshape(row, col)
    # Five stencil laplace filter
    laplace_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # Nine stencil laplace filter
    # laplace_filter = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    # smoothed_sol = sg.convolve2d(solution, -laplace_filter, mode="same")
    smoothed_sol = np.log(np.where(solution > 1e-6, solution, 1e-6))
    gradx, grady = np.gradient(smoothed_sol)

    fig, ax = plt.subplots(1, 4, figsize=(12, 4))

    ax[0].imshow(source, cmap="inferno")
    ax[0].set(title="source")
    ax[1].imshow(laplace.toarray(), cmap="inferno")
    ax[1].set(title="laplace matrix")
    ax[2].imshow(solution, cmap="inferno")
    ax[2].set(title="solution")
    ax[3].imshow(smoothed_sol, cmap="inferno")
    ax[3].quiver(grady, gradx, units="xy", angles="xy", scale=1)
    ax[3].set(title="Solution in log scale")
    plt.show()


C0 = ti.Vector([0, 0, 0])
C1 = ti.Vector([0, 1, 0])


@ti.func
def laplacian(field, i: int, j: int):
    return (
        field[i + 1, j]
        + field[i - 1, j]
        + field[i, j + 1]
        + field[i, j - 1]
        - 4.0 * field[i, j]
    )


@ti.data_oriented
class ExplicitDiffusion:
    def __init__(
        self,
        initial_concentration: np.ndarray,
        obstacles: np.ndarray,
        diffusivity: float = 1,
        dt: float = 1,
        eps: float = 1e-5,
    ) -> None:
        """_summary_

        Args:
            initial_concentration (np.ndarray): Map of values that represent the initial concentration of the diffusing substance
            obstacles (np.ndarray): Binary map of obstacles
            diffusivity (float, optional): Diffusivity coefficient of the simulation. Defaults to 1.
            dt (float, optional): Step size in time. Defaults to 1.
        """
        self.width, self.height = initial_concentration.shape
        self.diffusivity = diffusivity
        self.dt = dt
        assert eps > 0
        self.eps = eps
        # Define initial concentration map in numpy
        assert initial_concentration.shape == obstacles.shape
        self._initial_concentration = ti.field(
            dtype=float, shape=initial_concentration.shape
        )
        self._initial_concentration.from_numpy(initial_concentration)

        # Define obstacle map in numpy
        self._obstacles = ti.field(dtype=float, shape=obstacles.shape)
        self._obstacles.from_numpy(obstacles)

        self._concentration = ti.field(
            dtype=ti.f32, shape=(2, *initial_concentration.shape)
        )
        self._pixel = ti.field(dtype=ti.f32, shape=initial_concentration.shape)

    @ti.kernel
    def reset(self):
        for i, j in ti.ndrange(self.width, self.height):
            self._concentration[0, i, j] = self._initial_concentration[i, j]
            # self._concentration[1, i, j] = self._initial_concentration[i, j]

    @property
    def concentration(self) -> ti.Field:
        return self._concentration.to_numpy()[1, :, :]

    @ti.kernel
    def update(self, phase: int, dt: float):
        # assert phase in (0,1)
        for i, j in ti.ndrange((1, self.width - 1), (1, self.height - 1)):
            if self._obstacles[i, j] == 1:
                self._concentration[phase, i, j] = 0
            else:
                center = self._concentration[1 - phase, i, j]
                laplacian = (
                    self._concentration[1 - phase, i - 1, j]
                    + self._concentration[1 - phase, i + 1, j]
                    + self._concentration[1 - phase, i, j - 1]
                    + self._concentration[1 - phase, i, j + 1]
                    - 4 * center
                )
                self._concentration[phase, i, j] = (
                    center
                    + dt * laplacian * self.diffusivity
                    + self._initial_concentration[i, j]
                )  # Add stationary condition
            # center = self._concentration[i, j][1 - phase]
            # laplacian = (
            #     self._concentration[i - 1, j][1 - phase]
            #     + self._concentration[i + 1, j][1 - phase]
            #     + self._concentration[i, j - 1][1 - phase]
            #     + self._concentration[i, j + 1][1 - phase]
            #     - 4 * center
            # )
            # self._concentration[i, j][phase] = (
            #     center
            #     + dt * laplacian * self.diffusivity
            #     + self._initial_concentration[i, j]
            # )  # Add stationary condition

    @ti.kernel
    def render(self):
        for i, j in ti.ndrange(self.width, self.height):
            self._pixel[i, j] = self._concentration[1, i, j]

    @ti.kernel
    def is_stationary(self) -> bool:
        is_stationary = True
        for i, j in ti.ndrange(self.width, self.height):
            if self._concentration[1, i, j] < self.eps and (self._obstacles[i, j] != 1):
                is_stationary = False
        return is_stationary

    def run(self):
        self.reset()

        gui = ti.GUI("Concentration evolution", res=(self.height, self.width))
        # Sets the window title and the resolution
        substeps = 2
        i = 0
        t = 0
        while not self.is_stationary():
            for i in range(substeps):
                self.update(i % 2, self.dt / substeps)
            t += 1
        print(f"final time : {t * self.dt}")
        self.render()
        gui.contour(self._pixel, normalize=True)
        gui.show()


def main2():
    from snake_ai.envs import SlotEnv
    import matplotlib.pyplot as plt

    ti.init(arch=ti.gpu, device_memory_GB=4)
    WIDTH, HEIGHT = 20, 20
    PIXEL = 10
    FPS = 60

    env = SlotEnv(width=WIDTH, height=HEIGHT, pixel=PIXEL)
    env.reset()
    initial_concentration = np.zeros(env.window_size, dtype=float)
    initial_concentration[
        env.goal.centerx : env.goal.centerx + PIXEL,
        env.goal.centery : env.goal.centery + PIXEL,
    ] = 1e3

    obstacles = np.zeros(env.window_size, dtype=float)
    for obs in env.obstacles:
        obstacles[obs.x : obs.x + obs.width, obs.y : obs.y + obs.height] = 1

    plt.show()
    solver = ExplicitDiffusion(
        initial_concentration, obstacles, diffusivity=1, dt=1 / 10
    )
    solver.run()
    fig, ax = plt.subplots(1, 1)
    ax.imshow(solver.concentration)
    # ax[1].imshow(obstacles)
    plt.show()


if __name__ == "__main__":
    main()
