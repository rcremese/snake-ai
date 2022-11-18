##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Tests for modules inside physim
 # @desc Created on 2022-11-10 2:28:05 pm
 # @copyright https://mit-license.org/
 #
import pygame
from snake_ai.physim.particle import Particle
from snake_ai.physim.diffusion_process import DiffusionProcess
from snake_ai.envs import SnakeClassicEnv, SnakeAI
import numpy as np

class TestParticle:
    initial_pos = (1, 1)
    radius = 1

    def test_move(self):
        particle = Particle(*self.initial_pos, self.radius)

        particle.move(1, -1)
        x_new, y_new = particle._position
        assert x_new == 2 or y_new == 0

    def test_reset(self):
        particle = Particle(*self.initial_pos, self.radius)

        particle.reset(5, 5)
        x, y = particle._position
        assert x == 5 and y == 5

    def test_grid_conversion(self):
        particle = Particle(0.2, 0.6, self.radius)

        x, y = particle.get_grid_position()
        assert x == 0 and y == 1

    def test_collision(self):
        particle = Particle(0, 0, self.radius)

        pixel = [10, 10]
        obstacle_1 = pygame.Rect(2, 2, *pixel)
        assert not particle.collide(obstacle_1)
        obstacle_2 = pygame.Rect(-1, -1, *pixel)
        assert particle.collide(obstacle_2)
        # Edge case where the rectangle touch the particle but the
        obstacle_3 = pygame.Rect(0.5, 0.5, *pixel)
        assert particle.collide(obstacle_3)

    def test_collision_list(self):
        particle = Particle(0, 0, self.radius)

        pixel = [10, 10]
        rect_1 = pygame.Rect(0, 0, *pixel)
        rect_2 = pygame.Rect(1, 1, *pixel)
        assert particle.collide_any([rect_1, rect_2])
        rect_3 = pygame.Rect(-2, -2, 1, 1)
        rect_4 = pygame.Rect(2, 2, 1, 1)
        assert not particle.collide_any([rect_3, rect_4])

    def test_update(self):
        particle = Particle(*self.initial_pos, self.radius)

        particle.update_position(5, 5)
        x, y = particle._position
        assert x == 5 and y==5

    def test_equality(self):
        particle_A = Particle(*self.initial_pos, self.radius)

        particle_B = Particle(*self.initial_pos, self.radius)
        assert particle_A == particle_B
        particle_C = Particle(5, 5, self.radius)
        assert particle_A != particle_C
        particle_D = Particle(*self.initial_pos, 2)
        assert particle_A != particle_D

    def test_is_inside(self):
        particle = Particle(0.5, 0.5, self.radius)

        rect_A = pygame.Rect(-1, -1, 10, 10)
        assert particle.is_inside(rect_A)
        rect_B = pygame.Rect(1, 1, 10, 10)
        assert not particle.is_inside(rect_B)
        rect_C = pygame.Rect(0, 0, 1, 1)
        assert not particle.is_inside(rect_C, center=False)
        assert particle.is_inside(rect_C, center=True)

class TestDiffusionProcess:
    pixel_size = 10
    obstacle_free_env = SnakeClassicEnv(width=5, height=5, nb_obstacles=0, pixel=pixel_size)
    obstacle_free_env.food = pygame.Rect(0, 0, pixel_size, pixel_size)
    obstacle_free_env.snake = SnakeAI(2 * pixel_size, 2 * pixel_size, pixel_size=pixel_size)

    diff_coef = 1
    t_max = 10
    radius = 1

    def test_reset(self):
        ## Test one particule
        diff_process = DiffusionProcess(self.obstacle_free_env, nb_particles=1, t_max=self.t_max, diff_coef=self.diff_coef, part_radius=self.radius)
        diff_process.seed()
        # initialise the particle
        diff_process.reset()
        assert diff_process.particles == [Particle(45, 5, self.radius)]
        assert np.array_equal(diff_process._positions, np.array([[45, 5]]))
        assert diff_process._collisions == [False]
        ## Test several particules
        nb_part = 10
        mult_diff_process = DiffusionProcess(self.obstacle_free_env, nb_particles=nb_part, t_max=self.t_max, diff_coef=self.diff_coef, part_radius=self.radius)
        mult_diff_process.seed()
        mult_diff_process.reset()
        assert mult_diff_process.particles == nb_part * [Particle(45, 5, self.radius)]
        assert np.array_equal(mult_diff_process._positions, np.repeat([[45, 5]], repeats=nb_part, axis=0))
        assert np.all(~mult_diff_process._collisions)

    def test_source_positionning(self):
        diff_process = DiffusionProcess(self.obstacle_free_env, nb_particles=1, t_max=self.t_max, diff_coef=self.diff_coef, part_radius=self.radius)
        diff_process.reset()
        # define new food position
        food_position = (2 * self.pixel_size, 3 * self.pixel_size)
        particule_position = (2.5 * self.pixel_size, 3.5 * self.pixel_size)
        diff_process.set_source_position(*food_position)
        assert diff_process.particles == [Particle(*particule_position, self.radius)]
        assert np.array_equal(diff_process._positions, np.array([particule_position]))
        assert diff_process._collisions == [False]

    def test_step(self):
        ## Test one particule
        diff_process = DiffusionProcess(self.obstacle_free_env, nb_particles=1, t_max=self.t_max, diff_coef=self.diff_coef, part_radius=self.radius)
        diff_process.seed()
        # initialise the particle
        diff_process.reset()
        diff_process.step()
        assert diff_process.particles == [Particle(45.97873798410574,7.240893199201458, self.radius)]
        assert diff_process._collisions == [False]
        ## Test several particules
        nb_part = 5
        mult_diff_process = DiffusionProcess(self.obstacle_free_env, nb_particles=nb_part, t_max=self.t_max, diff_coef=self.diff_coef, part_radius=self.radius)
        mult_diff_process.seed()
        mult_diff_process.reset()
        mult_diff_process.step()
        assert mult_diff_process.particles == [
            Particle(45.97873798410574,7.240893199201458, self.radius),
            Particle(46.86755799014997,4.022722120123589, self.radius),
            Particle(45.95008841752559,4.848642791702302, self.radius),
            Particle(44.89678114820644,5.410598501938372, self.radius),
            Particle(45.14404357116088,6.454273506962975, self.radius),
            ]
        assert list(mult_diff_process._collisions) == nb_part * [False]

    def test_simulation(self):
        diff_process = DiffusionProcess(self.obstacle_free_env, nb_particles=5, t_max=self.t_max, diff_coef=self.diff_coef, part_radius=self.radius)
        diff_process.seed()
        # initialise the particle
        diff_process.reset()
        # 5 particules make 10 random steps inside the environment
        diff_process.start_simulation()
        assert diff_process.particles == [
            Particle(45.29126767451755,0.9123978292107418, 1),
            Particle(49.581175846883006,2.9020307728990913, 1),
            Particle(46.22222264677815,8.228521029121612, 1),
            Particle(41.94438988025139,6.7126291641622124, 1),
            Particle(41.17343567846424,6.997881428096835, 1)
        ]
        assert list(diff_process._collisions) == [True, True, False, False, False]