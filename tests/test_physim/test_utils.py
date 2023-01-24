import numpy as np
import pytest

from snake_ai.physim.regular_grid import RegularGrid2D
class TestRegularGrid:
    x_init, x_end, x_step = 0, 10, 2
    y_init, y_end, y_step = -3.5, 12, 3
    z_init, z_end, z_step = 42, 37, -0.5

    def test_init(self):
        reg_grid = RegularGrid2D(self.x_init, self.x_end, self.x_step, self.y_init, self.y_end, self.y_step)
        assert np.array_equal(reg_grid.x, [0, 2, 4, 6, 8])
        assert np.array_equal(reg_grid.y, [-3.5, -0.5, 2.5, 5.5, 8.5, 11.5])

        reg_grid_2 = RegularGrid2D(self.x_init, self.x_end)
        assert np.array_equal(reg_grid_2.x, reg_grid_2.y)
        assert reg_grid_2.x_step == 1 and reg_grid_2.y_step == 1

        reg_grid_3 = RegularGrid2D(self.z_init, self.z_end, self.z_step)
        assert np.array_equal(reg_grid_3.x, [42, 41.5, 41, 40.5, 40, 39.5, 39, 38.5, 38, 37.5])

    def test_factory(self):
        reg_grid = RegularGrid2D.regular_square(self.y_end)
        assert np.array_equal(reg_grid.x, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        reg_grid_2 = RegularGrid2D.regular_square(self.y_init, self.y_end)
        assert np.array_equal(reg_grid_2.y, [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5])
        reg_grid_3 = RegularGrid2D.regular_square(self.y_init, self.y_end, self.y_step)
        assert np.array_equal(reg_grid_3.x, [-3.5, -0.5, 2.5, 5.5, 8.5, 11.5])
        # Test errors
        with pytest.raises(LookupError):
            RegularGrid2D.regular_square(1, 2, 3, 4)
        with pytest.raises(AssertionError):
            RegularGrid2D.regular_square(-1.)

    def test_grid(self):
        reg_grid = RegularGrid2D(self.x_init, self.x_end, self.x_step, self.y_init, self.y_end, self.y_step)
        X, Y = reg_grid.mesh
        assert np.array_equal(X, np.repeat([[0, 2, 4, 6, 8]], 6, axis=0))
        assert np.array_equal(Y, np.repeat([[-3.5, -0.5, 2.5, 5.5, 8.5, 11.5]], 5, axis=0).T)

    def test_convolution_downgrade(self):
        reg_grid = RegularGrid2D(self.x_init, self.x_end, self.x_step, self.y_init, self.y_end, self.y_step)
        conv_window = ConvolutionWindow.gaussian(3)
        reg_grid.convolution_dowgrade(conv_window, stride = 1, mode='valid')
        assert np.array_equal(reg_grid.x, [2, 4, 6, 8])
        assert np.array_equal(reg_grid.y, [-0.5, 2.5, 5.5, 8.5, 11.5])
        assert (reg_grid.x_step == 2) and (reg_grid.y_step == 3)
        reg_grid.convolution_dowgrade(conv_window, stride=2, mode='valid')
        assert np.array_equal(reg_grid.x, [4, 8])
        assert np.array_equal(reg_grid.y, [2.5, 8.5])
        reg_grid.convolution_dowgrade(conv_window, stride=1, mode='same')
        assert np.array_equal(reg_grid.x, [4, 8])
        assert np.array_equal(reg_grid.y, [2.5, 8.5])





from snake_ai.physim.convolution_window import ConvolutionWindow, exp_2d
class TestConvolutionWindow:
    size = 10

    def test_init(self):
        conv_window = ConvolutionWindow(self.size, 'Gaussian')
        assert conv_window.size == self.size
        assert conv_window.conv_type == 'gaussian'
        assert conv_window._conv_window is None
        assert conv_window.std == 1
        mean_window = ConvolutionWindow(6.3, 'Mean')
        assert mean_window.size == 6
        assert mean_window.conv_type == 'mean'
        dx_window = ConvolutionWindow(self.size, 'Gaussian_DX', 9.5)
        assert dx_window.conv_type == 'gaussian_dx'
        assert dx_window.std == 9.5
        dy_window = ConvolutionWindow(self.size, 'Gaussian_Dy')
        assert dy_window.conv_type == 'gaussian_dy'
        assert dy_window.std == 1

        # Errors
        with pytest.raises(ValueError):
            ConvolutionWindow(-12, 'mean')
        with pytest.raises(ValueError):
            ConvolutionWindow(self.size, 'test')
        with pytest.raises(ValueError):
            ConvolutionWindow(self.size, 'gaussian', -5)


    def test_factories(self):
        mean_window = ConvolutionWindow.mean(self.size)
        assert mean_window.conv_type == 'mean'
        assert mean_window.size == self.size
        assert mean_window._conv_window is None
        gauss_window = ConvolutionWindow.gaussian(self.size, 1)
        assert gauss_window.conv_type =='gaussian'
        assert gauss_window.std == 1
        gauss_window_2 = ConvolutionWindow.gaussian(self.size)
        assert gauss_window_2.std == self.size / 6
        gauss_dx = ConvolutionWindow.gaussian_dx(self.size)
        assert gauss_dx.conv_type == 'gaussian_dx'
        assert gauss_dx.size == self.size
        gauss_dy = ConvolutionWindow.gaussian_dy(self.size)
        assert gauss_dy.conv_type == 'gaussian_dy'
        assert gauss_dy.size == self.size


    def test_value(self):
        mean_window = ConvolutionWindow(self.size, 'mean')
        assert np.array_equal(mean_window.value, np.ones((self.size, self.size)) / self.size ** 2)

        space = np.linspace(- 0.5 * self.size, 0.5 * self.size, self.size)
        X, Y = np.meshgrid(space, space)
        gaussian = exp_2d(X, Y)

        gauss_window = ConvolutionWindow(self.size, 'gaussian', std=1)
        assert np.array_equal(gauss_window.value, gaussian)
        gauss_dx = ConvolutionWindow(self.size, 'gaussian_dx', std=5)
        gaussian_5 = exp_2d(X, Y, std=5)
        assert np.array_equal(gauss_dx.value, -X * gaussian_5 / 25)
        gauss_dy = ConvolutionWindow.gaussian_dy(self.size)
        gaussian_3sigma = exp_2d(X, Y, std=gauss_dy.std)
        assert np.array_equal(gauss_dy.value, -Y * gaussian_3sigma / gauss_dy.std**2)

from snake_ai.physim.gradient_field import GradientField
from snake_ai.physim.walker import Walker
class TestWalker:
    grid_space = np.arange(0.5, 10.5)

    X, Y = np.meshgrid(grid_space, grid_space, indexing='ij')
    gradient_field = GradientField(np.zeros((10,10)))
    gradient_field._gradient_map = 5 + np.stack([-X, -Y])
    init_pos = [0.5, 6.5]

    def test_step(self):
        walker = Walker(self.init_pos)
        walker.step(self.gradient_field)
        assert np.array_equal(walker.position, [5, 5])
        assert walker.time == 1