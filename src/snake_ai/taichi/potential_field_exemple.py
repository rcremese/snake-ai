from snake_ai.envs import GridWorld, RandomObstaclesEnv, EnvConverter
from snake_ai.taichi.boxes import convert_rectangles, Box2D

import taichi as ti
import taichi.math as tm
import numpy as np
import logging


@ti.data_oriented
class ArtificialPotentialField:
    def __init__(self, env: GridWorld, init_pos: np.ndarray, max_range: float) -> None:
        assert isinstance(init_pos, np.ndarray)
        assert init_pos.ndim == 2 and init_pos.shape[1] == 2
        assert isinstance(env, GridWorld)
        self.env = env
        self.nb_walker = init_pos.shape[0]
        self._init_pos = init_pos
        self.pos = ti.Vector.field(
            2, dtype=ti.f32, shape=self.nb_walker, needs_grad=True
        )  # particle positions
        self.vel = ti.Vector.field(
            2, dtype=ti.f32, shape=self.nb_walker
        )  # particle velocities
        self.goal = tm.vec2(*env.goal.center)
        assert max_range > 0, "Only positive values allowed for max_range"
        self.max_range = float(max_range)
        # Save the obstacles
        self.nb_obstacles = env.nb_obstacles
        self.obstacles = convert_rectangles(env.obstacles)
        logging.debug(
            f"Nb obstacles : {self.nb_obstacles}, Obstacles: {self.obstacles}, shape = {self.obstacles.shape}"
        )
        # Diffine the potential fields
        self.atractive_field = ti.field(
            dtype=ti.f32, shape=(self.nb_walker), needs_grad=True
        )  # potential energy
        self.repulsive_field = ti.field(
            dtype=ti.f32, shape=(self.nb_walker, self.nb_obstacles), needs_grad=True
        )
        self.total_field = ti.field(
            dtype=ti.f32, shape=(self.nb_walker), needs_grad=True
        )
        # Define the weights associated with it
        self.atractive_weight = ti.field(dtype=ti.float32, shape=(), needs_grad=True)
        self.atractive_weight[None] = 1.0
        self.repulsive_weights = ti.field(
            dtype=ti.float32, shape=(self.nb_obstacles), needs_grad=True
        )
        self.repulsive_weights.from_numpy(np.ones(self.nb_obstacles))

    def reset(self):
        self.pos.from_numpy(self._init_pos)
        self.vel.from_numpy(np.zeros((self.nb_walker, 2)))

    @ti.kernel
    def init_grad(self):
        for walker, obs in ti.ndrange(self.nb_walker, self.nb_obstacles):
            self.atractive_field.grad[walker] = 1.0
            self.repulsive_field.grad[walker, obs] = 0.0
            self.total_field.grad[walker] = 1.0
            self.repulsive_weights.grad[obs] = 1.0
            self.pos.grad[walker] = 0.0
        self.atractive_weight.grad[None] = 1.0

    @ti.kernel
    def evaluate_attractive_field(self):
        for walker in ti.ndrange(self.nb_walker):
            self.atractive_field[walker] = (
                0.5 * tm.length(self.pos[walker] - self.goal) ** 2
            )

    @ti.kernel
    def evaluate_repulsive_field(self):
        for walker, obs in ti.ndrange(self.nb_walker, self.nb_obstacles):
            dist = self.distance(self.pos[walker], self.obstacles[obs])
            if dist > self.max_range or dist <= 0:
                self.repulsive_field[walker, obs] = 0
            else:
                self.repulsive_field[walker, obs] = (
                    0.5 * ((self.max_range - dist) / (self.max_range * dist)) ** 2
                )

    @ti.kernel
    def evaluate_total_field(self):
        for walker, obs in ti.ndrange(self.nb_walker, self.nb_obstacles):
            self.total_field[walker] += (
                self.atractive_weight[None] * self.atractive_field[walker]
            ) / self.nb_obstacles + (
                self.repulsive_weights[obs] * self.repulsive_field[walker, obs]
            )

    @staticmethod
    @ti.func
    def distance(position: tm.vec2, obstacle: Box2D) -> float:
        dist_vector = tm.max(obstacle.min - position, 0, position - obstacle.max)
        return tm.length(dist_vector)

    def potential_field_evaluation(self):
        self.init_grad()
        self.evaluate_attractive_field()
        self.evaluate_repulsive_field()
        self.evaluate_total_field()
        self.evaluate_total_field.grad()
        logging.debug(
            f"Potential field: {self.total_field}, grad: {self.total_field.grad}"
        )

        self.evaluate_attractive_field.grad()
        logging.debug(
            f"Attractive field: {self.atractive_field}, grad: {self.atractive_field.grad}"
        )
        self.evaluate_repulsive_field.grad()
        logging.debug(
            f"Repulsive field: {self.repulsive_field}, grad: {self.repulsive_field.grad}"
        )

    @ti.kernel
    def step(self, dt: float):
        for walker in ti.ndrange(self.nb_walker):
            self.pos[walker] -= dt * self.pos.grad[walker]
            # self.pos[walker] += dt * self.vel[walker]

    def run(self, dt: float = 1):
        pixel = 20
        scale_vector = np.array((pixel * env.width, pixel * env.height))

        gui = ti.GUI("Autodiff gravity", scale_vector)
        self.reset()
        while gui.running:
            self.potential_field_evaluation()
            self.step(dt)
            # Plot the potential field
            centers = self.pos.to_numpy() / pixel
            gui.circles(centers, radius=3)
            for obs in self.env.obstacles:
                gui.rect(obs.min / pixel, obs.max / pixel, color=0xFF0000)
            # gui.circles(self.pos.to_numpy() / scale_vector, radius=3)
            gui.show()


def example():
    ti.init(arch=ti.gpu)

    N = 50
    dt = 1e-5

    x = ti.Vector.field(2, dtype=ti.f32, shape=N, needs_grad=True)  # particle positions
    v = ti.Vector.field(2, dtype=ti.f32, shape=N)  # particle velocities
    U = ti.field(dtype=ti.f32, shape=(), needs_grad=True)  # potential energy

    @ti.kernel
    def compute_U():
        for i, j in ti.ndrange(N, N):
            r = x[i] - x[j]
            # r.norm(1e-3) is equivalent to ti.sqrt(r.norm()**2 + 1e-3)
            # This is to prevent 1/0 error which can cause wrong derivative
            U[None] += -1 / r.norm(1e-3)  # U += -1 / |r|

    @ti.kernel
    def advance():
        for i in x:
            v[i] += dt * -x.grad[i]  # dv/dt = -dU/dx
        for i in x:
            x[i] += dt * v[i]  # dx/dt = v

    def substep():
        with ti.ad.Tape(loss=U):
            # Kernel invocations in this scope will later contribute to partial derivatives of
            # U with respect to input variables such as x.
            compute_U()  # The tape will automatically compute dU/dx and save the results in x.grad
        advance()

    @ti.kernel
    def init():
        for i in x:
            x[i] = [ti.random(), ti.random()]

    init()
    gui = ti.GUI("Autodiff gravity")
    while gui.running:
        for i in range(50):
            substep()
        gui.circles(x.to_numpy(), radius=3)
        gui.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    env = RandomObstaclesEnv(20, 20, pixel=1, nb_obs=10, max_obs_size=5)
    env.reset()
    converter = EnvConverter(env)
    init_pos = converter.get_agent_position()
    init_pos = converter.convert_free_positions_to_point_cloud(step=2)
    logging.debug("Initial position: %s", init_pos)
    logging.debug("Goal position: %s", converter.get_goal_position())

    ti.init(arch=ti.gpu)
    simu = ArtificialPotentialField(env, init_pos=init_pos, max_range=10)
    simu.run(dt=0.01)
