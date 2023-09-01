import taichi as ti
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import taichi.math as tm
from typing import Dict, List, Tuple

GAMMA = 10


@ti.func
def lerp(a, b, t):
    t = tm.clamp(t, 0, 1)
    return a + (b - a) * t


@ti.dataclass
class Particle:
    pos: ti.math.vec2
    vel: ti.math.vec2
    radius: ti.f32


@ti.dataclass
class Box:
    min: tm.vec2
    max: tm.vec2

    @ti.func
    def width(self):
        return self.max[0] - self.min[0]

    @ti.func
    def height(self):
        return self.max[1] - self.min[1]

    @ti.func
    def is_inside(self, pos: ti.math.vec2) -> bool:
        clamped_pos = tm.clamp(pos, self.min, self.max)
        return clamped_pos.x == pos.x and clamped_pos.y == pos.y


@ti.dataclass
class Rectangle:
    upper_left: ti.math.vec2
    width: int
    height: int

    @ti.func
    def is_outside(self, pos: ti.math.vec2):
        return (
            (pos[0] < self.upper_left[0])
            or (pos[0] > self.upper_left[0] + self.width)
            or (pos[1] < self.upper_left[1])
            or (pos[1] > self.upper_left[1] + self.height)
        )


@ti.dataclass
class State:
    pos: tm.vec2
    vel: tm.vec2


@ti.data_oriented
class DifferentiableSimulation:
    def __init__(
        self,
        particles: Particle,
        force_field: ti.Field,
        t_max: float,
        dt: float,
        diffusivity: float,
        bounds: Rectangle,
        target: tm.vec2,
        max_epoch: int,
        lr: float,
        obstacles: Rectangle = None,
    ) -> None:
        ## TODO : type check
        self._particles = particles
        self.nb_particles = particles.shape[0]

        self._obstacles = obstacles
        self.nb_obstacles = obstacles.shape[0]

        self.nb_steps = int(t_max / dt)
        self._states = State.field(
            shape=(self.nb_particles, self.nb_steps), needs_grad=True
        )
        # Stochasticity
        noise = np.random.randn(self.nb_particles, self.nb_steps, 2).astype(np.float32)
        self.diffusivity = diffusivity
        self._noise = ti.Vector.field(
            2, dtype=ti.f32, shape=(self.nb_particles, self.nb_steps)
        )
        self._noise.from_numpy(noise)

        self.bounds = bounds
        self.width = bounds.width
        self.height = bounds.height
        # ti.Vector.field(2, dtype=ti.f32, shape=(self._nb_particles, int(t_max / dt)))
        self._collision_count = ti.field(ti.i32, shape=(self.nb_particles))

        self.force_field = force_field
        self.t_max = t_max
        self.dt = dt

        self.target = target
        self.loss = ti.field(ti.f32, shape=(), needs_grad=True)
        # training parameters
        self.max_epoch = max_epoch
        self.lr = lr

    @ti.kernel
    def reset(self):
        for n in self._particles:
            self._states[n, 0].pos = self._particles[n].pos
            self._states[n, 0].vel = self._particles[n].vel
            self._collision_count[n] = 0

    @ti.kernel
    def step(self, t: int):
        for n in range(self.nb_particles):
            i, j = ti.floor(self._states[n, t - 1].pos).cast(int)
            # TODO : remove this shit !
            if i < 0 or i >= self.width or j < 0 or j >= self.height:
                self._states[n, t].pos = self._states[n, t - 1].pos
            else:
                self._states[n, t].pos = (
                    self._states[n, t - 1].pos
                    + self.dt * self.force_field[i, j]
                    + tm.sqrt(2 * self.dt * self.diffusivity) * self._noise[n, t]
                )
            # self._states[n, t].vel = (
            #     self._states[n, t - 1].vel + self.dt * self.force_field[i, j]
            # )
            # self._states[n, t].pos = (
            #     self._states[n, t - 1].pos + self.dt * self._states[n, t].vel
            # )
            if self.bounds.is_outside(self._states[n, t].pos):
                pos, vel = self.collision(
                    self._states[n, t].pos, self._states[n, t].vel
                )
                self._states[n, t].pos = pos
                self._states[n, t].vel = vel

            for o in range(self.nb_obstacles):
                if self._obstacles[o].is_inside(self._states[n, t].pos):
                    self.obstacle_collision(t, n, o)
                    self._collision_count[n] += 1

            # self._states[n, t].vel = self._states[n, t - 1].vel + self.dt * self.force_field[i,j]
            # self._states[n, t].pos = self._states[n, t - 1].pos + self.dt * self._states[n, t].vel
            # self._states[n, t].pos = self._states[n, t - 1].pos + self.dt * self.force_field[i,j]

    @ti.func
    def collision(self, position: tm.vec2, velocity: tm.vec2) -> Tuple[tm.vec2]:
        new_pos, new_vel = position, velocity
        if position.x < 0:
            new_pos = tm.vec2(0, position.y)
            new_vel = tm.vec2(-velocity.x, velocity.y)
        if position.x > self.height:
            new_pos = tm.vec2(self.height, position.y)
            new_vel = tm.vec2(-velocity.x, velocity.y)
        if position.y < 0:
            new_pos = tm.vec2(position.x, 0)
            new_vel = tm.vec2(velocity.x, -velocity.y)
        if position.y > self.width:
            new_pos = tm.vec2(position.x, self.width)
            new_vel = tm.vec2(velocity.x, -velocity.y)
        return new_pos, new_vel

    def run(self):
        self.reset()
        for t in range(1, self.nb_steps):
            self.step(t)
        self.compute_loss(t)

    @ti.kernel
    def compute_loss(self, t: int):
        for n in range(self.nb_particles):
            self.loss[None] += (
                tm.length(self._states[n, t].pos - self.target)
                # + GAMMA * self._collision_count[n]
            ) / self.nb_particles

    def optimize(self):
        for iter in range(self.max_epoch):
            with ti.ad.Tape(self.loss):
                self.run()
            print("Iter=", iter, "Loss=", self.loss[None])
            self._update_force_field()

    @ti.kernel
    def _update_force_field(self):
        for i, j in self.force_field:
            self.force_field[i, j] -= self.lr * self.force_field.grad[i, j]
            ## Gradient clipping
            # if tm.length(self.force_field[i, j]) > 1:
            #     self.force_field[i, j] = tm.normalize(self.force_field[i, j])
        ## Setting the force field to 0 inside obstacles
        # inside_obs = False
        # for obs in range(self.nb_obstacles):
        #     if self._obstacles[obs].is_inside(tm.vec2(i, j)):
        #         inside_obs = True
        # if inside_obs:
        #     self.force_field[i, j] = tm.vec2(0, 0)
        # else:
        #     self.force_field[i, j] -= self.lr * self.force_field.grad[i, j]

    @ti.func
    def obstacle_collision(self, t: int, n: int, o: int):
        for i in ti.static(range(2)):
            # Collision on the min border
            if (self._states[n, t - 1].pos[i] < self._obstacles[o].min[i]) and (
                self._states[n, t].pos[i] >= self._obstacles[o].min[i]
            ):
                # Calculate the time of impact
                toi = (self._obstacles[o].min[i] - self._states[n, t - 1].pos[i]) / (
                    self._states[n, t].pos[i] - self._states[n, t - 1].pos[i]
                )
                self._states[n, t].pos[i] = lerp(
                    self._states[n, t - 1].pos[i], self._states[n, t].pos[i], toi
                ) - ti.abs(self._obstacles[o].min[i] - self._states[n, t].pos[i])
                # self._states[n, t].pos[i] = self._obstacles[o].min[i] - ti.abs(
                #     self._obstacles[o].min[i] - self._states[n, t].pos[i]
                # )
                # Reflect velocity
                self._states[n, t].vel[i] = -self._states[n, t].vel[i]
            # Collision on the max border
            elif (self._states[n, t - 1].pos[i] > self._obstacles[o].max[i]) and (
                self._states[n, t].pos[i] <= self._obstacles[o].max[i]
            ):
                # Calculate the time of impact
                toi = (self._obstacles[o].max[i] - self._states[n, t - 1].pos[i]) / (
                    self._states[n, t].pos[i] - self._states[n, t - 1].pos[i]
                )
                self._states[n, t].pos[i] = lerp(
                    self._states[n, t - 1].pos[i], self._states[n, t].pos[i], toi
                ) + ti.abs(self._obstacles[o].max[i] - self._states[n, t].pos[i])

                # self._states[n, t].pos[i] = self._obstacles[o].max[i] + ti.abs(
                #     self._obstacles[o].max[i] - self._states[n, t].pos[i]
                # )
                # Reflect velocity
                self._states[n, t].vel[i] = -self._states[n, t].vel[i]

        # LEGACY : Update x position and velocity
        # if (
        #     self._states[part_idx, time_idx - 1].pos.x < self._obstacles[obs_idx].min.x
        #     and self._states[part_idx, time_idx].pos.x >= self._obstacles[obs_idx].min.x
        # ):
        #     self._states[part_idx, time_idx].pos.x = self._obstacles[obs_idx].min.x
        #     self._states[part_idx, time_idx].vel.x = -self._states[
        #         part_idx, time_idx
        #     ].vel.x
        # elif (
        #     self._states[part_idx, time_idx - 1].pos.x > self._obstacles[obs_idx].max.x
        #     and self._states[part_idx, time_idx].pos.x <= self._obstacles[obs_idx].max.x
        # ):
        #     self._states[part_idx, time_idx].pos.x = self._obstacles[obs_idx].max.x
        #     self._states[part_idx, time_idx].vel.x = -self._states[
        #         part_idx, time_idx
        #     ].vel.x

    @property
    def particles(self):
        return self._particles.to_numpy()

    @property
    def trajectories(self) -> Dict[str, np.ndarray]:
        return self._states.to_numpy()


def render(
    simulation: DifferentiableSimulation,
    concentration: np.ndarray,
    cmap: str = "inferno",
    output_dir: Path = None,
):
    gui = ti.GUI("Differentiable Simulation", (simulation.width, simulation.height))

    trajectories = simulation.trajectories

    obstacles = simulation._obstacles.to_numpy()
    if output_dir:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(exist_ok=True)
    filename = None

    for t in range(simulation.nb_steps):
        gui.contour(concentration, normalize=True)
        scale_vector = np.array([simulation.width, simulation.height])
        pos = trajectories["pos"][:, t] / scale_vector
        gui.circles(pos, radius=5, color=int(ti.rgb_to_hex([255, 0, 0])))
        # # Obstacle
        # for obs_min, obs_max in zip(obstacles["min"], obstacles["max"]):
        #     gui.rect(
        #         obs_min / scale_vector,
        #         obs_max / scale_vector,
        #         radius=5,
        #         color=0x0000FF,
        #     )
        if output_dir:
            filename = output_dir.joinpath(f"frame_{t:04d}.png").as_posix()
        gui.show(filename)


def main():
    from snake_ai.utils.io import SimulationLoader
    from snake_ai.phiflow import maths
    from phi import flow
    import time

    ti.init(debug=True)

    sim_dir = Path("/home/rcremese/projects/snake-ai/simulations").resolve(strict=True)
    field_path = sim_dir.joinpath("Slot(20,20)_pixel_Tmax=800.0_D=1", "seed_0")
    # field_path = sim_dir.joinpath(
    #     "RandomObstacles(20,20)_pixel_Tmax=800.0_D=1", "seed_0"
    # )
    loader = SimulationLoader(field_path)
    simu = loader.load()

    # Define the concentration field and its gradient
    concentration_field = simu.field
    log_concentration = maths.compute_log_concentration(concentration_field)
    np_field = log_concentration.values.numpy("x,y")
    dx_concentration = flow.field.spatial_gradient(log_concentration)
    dx_concentration = maths.clip_gradient_norm(dx_concentration, 10)
    np_force_field = dx_concentration.values.numpy("x,y,vector")
    width, height = np_field.shape

    concentration = ti.field(ti.f32, shape=(width, height))
    concentration.from_numpy(np_field)

    force_field = ti.Vector.field(
        2, dtype=ti.f32, shape=(width, height), needs_grad=True
    )
    force_field.from_numpy(np_force_field)

    np_pts_cloud = simu.point_cloud.points.numpy("walker,vector")
    # Declares a field of particles and initialize it with the point cloud positions
    point_cloud = Particle.field(shape=(np_pts_cloud.shape[0],), needs_grad=True)
    point_cloud.pos.from_numpy(np_pts_cloud * 10)
    # Define objectives and bounds
    bounds = Rectangle(tm.vec2([0.0, 0.0]), width, height)
    target = tm.vec2(simu.env.goal.center)
    # Define the obstacles
    obstacles = Box.field(shape=(len(simu.env.obstacles)), needs_grad=True)
    obs_min = np.array([obs.topleft for obs in simu.env.obstacles])
    obs_max = np.array([obs.bottomright for obs in simu.env.obstacles])
    obstacles.min.from_numpy(obs_min)
    obstacles.max.from_numpy(obs_max)
    # simulation = ParticleSimulation(point_cloud, force_field, 1000, 1, bounds, target)
    simulation = DifferentiableSimulation(
        point_cloud,
        force_field,
        t_max=400,
        dt=1,
        obstacles=obstacles,
        bounds=bounds,
        target=target,
        diffusivity=0.1,
        max_epoch=100,
        lr=1,
    )
    tic = time.perf_counter()
    simulation.run()
    toc = time.perf_counter()
    print(f"Simulation time : {toc -tic} s")
    # render(
    #     simulation,
    #     concentration,
    #     output_dir=field_path.joinpath("initial_walkers").as_posix(),
    # )
    ## Optimization step
    tic = time.perf_counter()
    simulation.optimize()
    toc = time.perf_counter()
    print(f"Optimization time : {toc -tic} s")

    render(
        simulation,
        concentration,
        # output_dir=field_path.joinpath("optimized_walkers").as_posix(),
    )

    trajectories = simulation.trajectories
    plt.imshow(
        concentration.to_numpy(), cmap="plasma"
    )  # , extent=[0, width, 0, height])
    plt.quiver(
        force_field.to_numpy()[:, :, 1],
        force_field.to_numpy()[:, :, 0],
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    plt.scatter(trajectories["pos"][:, 0, 1], trajectories["pos"][:, 0, 0], c="b")
    plt.scatter(trajectories["pos"][:, -1, 1], trajectories["pos"][:, -1, 0], c="r")
    plt.scatter(target[1], target[0], c="k", marker="x")
    plt.show()


if __name__ == "__main__":
    main()
