##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Tests for modules inside physim
 # @desc Created on 2022-11-10 2:28:05 pm
 # @copyright https://mit-license.org/
 #
import pygame
from snake_ai.physim.particle import Particle
from snake_ai.physim.convolution_window import ConvolutionWindow
from snake_ai.physim.diffusion_process import DiffusionProcess
from snake_ai.envs import SnakeClassicEnv, SnakeAI
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest

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
    """Class to test a diffusion process

    The environment looks like that :
    O---F
    -----  O : obstacle
    SSS--  F : food
    -O---  S : snake
    -----  - : empty pixel
    """
    pixel_size = 10
    env = SnakeClassicEnv(width=5, height=5, nb_obstacles=0, pixel=pixel_size)
    env.seed(0)
    env.reset()
    # control environment
    food = pygame.Rect(4 * pixel_size, 0, pixel_size, pixel_size)
    snake = SnakeAI(2 * pixel_size, 2 * pixel_size, pixel_size=pixel_size)
    obstacles = [pygame.Rect(0, 0, pixel_size, pixel_size), pygame.Rect(pixel_size, 3 * pixel_size, pixel_size, pixel_size)]
    positions= np.array([
            [0.5, 0.5], # collide
            [2 * pixel_size, pixel_size],
            [1.5 * pixel_size, 3.5 * pixel_size], # collide
            [2 * pixel_size + 0.5, pixel_size + 0.8],
            [4 * pixel_size + 0.1, 3 * pixel_size + 0.6],
        ])


    diff_coef = 1
    window_size = env.window_size
    t_max = 10
    radius = 1
    nb_part = 5
    seed = 42

    def test_reset(self):
        ## Test one particule
        diff_process = DiffusionProcess(nb_particles=1, window_size=self.window_size, t_max=self.t_max, diff_coef=self.diff_coef, part_radius=self.radius, seed=self.seed)
        # initialise the particle
        diff_process.reset(*self.env.food.center)
        assert diff_process.particles == [Particle(45, 5, self.radius)]
        assert np.array_equal(diff_process._positions, np.array([[45, 5]]))
        assert diff_process._collisions.tolist() == [False]
        ## Test several particules
        nb_part = 10
        mult_diff_process = DiffusionProcess(nb_particles=nb_part, window_size=self.window_size, t_max=self.t_max, diff_coef=self.diff_coef, part_radius=self.radius)
        mult_diff_process.reset(*self.env.food.center)
        assert mult_diff_process.particles == nb_part * [Particle(45, 5, self.radius)]
        assert np.array_equal(mult_diff_process._positions, np.repeat([[45, 5]], repeats=nb_part, axis=0))
        assert np.all(~mult_diff_process._collisions)

    def test_source_positionning(self):
        diff_process = DiffusionProcess(nb_particles=1, window_size=self.window_size, t_max=self.t_max, diff_coef=self.diff_coef, part_radius=self.radius)
        diff_process.reset(*self.env.food.center)
        # define new food position
        particule_position = (2.5 * self.pixel_size, 3.5 * self.pixel_size)
        diff_process._init_particles(*particule_position)
        assert diff_process.particles == [Particle(*particule_position, self.radius)]
        assert np.array_equal(diff_process._positions, np.array([particule_position]))
        assert diff_process._collisions.tolist() == [False]

    def test_step(self):
        ## Test one particule
        diff_process = DiffusionProcess(nb_particles=1, window_size=self.window_size, t_max=self.t_max, diff_coef=self.diff_coef, part_radius=self.radius)
        # initialise the particle
        diff_process.reset(*self.env.food.center)
        diff_process.step()
        assert diff_process.particles == [Particle(45.193077087402344,4.473217010498047, self.radius)]
        assert diff_process._collisions == np.array([False])
        ## Test several particules
        mult_diff_process = DiffusionProcess(nb_particles=self.nb_part, window_size=self.window_size, t_max=self.t_max, diff_coef=self.diff_coef, part_radius=self.radius)
        mult_diff_process.reset(*self.env.food.center)
        mult_diff_process.step()
        assert mult_diff_process.particles == [
            Particle(44.611873626708984,4.955128192901611, self.radius),
            Particle(42.957275390625,5.0793232917785645, self.radius),
            Particle(45.333499908447266,5.795997619628906, self.radius),
            Particle(43.55880355834961,3.307002067565918, self.radius),
            Particle(44.62630844116211,3.459886074066162, self.radius),
            ]
        assert list(mult_diff_process._collisions) == self.nb_part * [False]

    def test_simulation(self):
        diff_process = DiffusionProcess(nb_particles=self.nb_part, window_size=self.window_size, t_max=self.t_max, diff_coef=self.diff_coef, part_radius=self.radius)
        # initialise the particle
        diff_process.reset(*self.env.food.center)
        # 5 particules make 10 random steps inside the environment
        diff_process.start_simulation()
        assert diff_process.particles == [
            Particle(44.296260833740234, 4.973932266235352, 1),
            Particle(41.42985534667969,9.147778511047363, 1),
            Particle(42.65422821044922,13.290007591247559, 1),
            Particle(45.85160446166992,3.1622676849365234, 1),
            Particle(44.04609680175781,3.6063549518585205, 1)
        ]
        assert list(diff_process._collisions) == 5 * [False, ]
        assert diff_process.time == self.t_max

    def test_collisions(self):
        diff_process = DiffusionProcess(nb_particles=self.nb_part, window_size=self.window_size, t_max=self.t_max, diff_coef=self.diff_coef, part_radius=self.radius, obstacles=self.obstacles)
        diff_process.reset(*self.env.food.center)
        diff_process.obstacles = self.obstacles
        assert diff_process.check_collisions(self.positions, diff_process.window_size, diff_process._obstacles).tolist() == [True, False, True, False, False]
        with pytest.raises(AssertionError):
            wrong_pos = np.array([0, 2])
            diff_process.check_collisions(wrong_pos, diff_process.window_size, diff_process._obstacles)

    def test_concentration_map(self):
        diff_process = DiffusionProcess(nb_particles=self.nb_part, window_size=self.window_size, t_max=self.t_max, diff_coef=self.diff_coef, part_radius=self.radius)
        diff_process.reset(*self.env.food.center)
        diff_process.obstacles = self.obstacles
        diff_process.positions = self.positions
        # test with 5 particles
        concentration_map = diff_process._compute_concentration_map()
        assert concentration_map[0,0] == 0
        assert concentration_map[2 * self.pixel_size, self.pixel_size] == 2
        assert concentration_map[int(1.5 * self.pixel_size), int(3.5 * self.pixel_size)] == 0
        assert concentration_map[4 * self.pixel_size, 3 * self.pixel_size] == 1
        assert concentration_map.sum() == 3
        # test with 5000 particles
        diff_process.nb_particles = 5_000
        diff_process.step()
        concentration_map = diff_process._accelerated_concentration_map()
        assert concentration_map.sum() == 5_000
class TestConvolutionWindow:
    pixel_size = 10

    def test_init(self):
        conv_window = ConvolutionWindow(self.pixel_size, 'gaussian')

        space = jnp.arange(- self.pixel_size / 2, self.pixel_size / 2)
        conv_1 = jsp.stats.norm.pdf(space) * jsp.stats.norm.pdf(space[:, None])
        assert jnp.isclose(conv_window.window, conv_1).all()

        std = 5
        conv_window = ConvolutionWindow(self.pixel_size, 'gaussian', std)

        conv_2 = jsp.stats.norm.pdf(space, scale=std) * jsp.stats.norm.pdf(space[:, None], scale=std)
        assert jnp.isclose(conv_window.window, conv_2).all()

        conv_window_mean = ConvolutionWindow(self.pixel_size, 'mean')
        conv_3 = np.ones((self.pixel_size, self.pixel_size)) / self.pixel_size**2
        assert jnp.isclose(conv_window_mean.window, conv_3).all()

        factory_gauss = ConvolutionWindow.gaussian(self.pixel_size, std)
        assert jnp.isclose(conv_window.window, factory_gauss).all()
        factory_mean = ConvolutionWindow.mean(self.pixel_size)
        assert jnp.isclose(conv_window_mean.window, factory_mean).all()