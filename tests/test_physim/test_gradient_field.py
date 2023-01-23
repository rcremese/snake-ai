from snake_ai.physim.gradient_field import GradientField
from snake_ai.physim.convolution_window import ConvolutionWindow
from snake_ai.physim.regular_grid import RegularGrid2D

import numpy as np

class TestSmoothGradientField:
    concentration_map = np.array([
        [0, 1, 2, 3, 2],
        [1, 2, 1, 2, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
    ])
    grid = RegularGrid2D.regular_square(5)

class TestGradientMap:
    concentration_map = np.array([
        [0, 1, 2, 3, 2],
        [1, 2, 1, 2, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
    ])
    mean_field = np.array([
        [8/9, 13/9, 12/9],
        [6/9, 8/9, 6/9],
        [5/9, 7/9, 5/9],
    ])
    # mean_field = np.array([
    #     # [4/9, 7/9, 11/9, 11/9, 8/9],
    #     [5/9, 8/9, 13/9, 12/9, 1],
    #     [4/9, 6/9, 8/9, 6/9, 4/9],
    #     [2/9, 5/9, 7/9, 5/9, 2/9],
    #     # [1/9, 4/9, 5/9, 4/9, 1/9],
    # ])
    conv_window = ConvolutionWindow.mean(3)
    eps = 1e-4

    def test_init(self):
        grad_field = GradientField(self.concentration_map, self.conv_window)
        assert all([np.array_equal(discret_space, np.linspace(0.5, 4.5, 5)) for discret_space in grad_field.discret_space])
        assert np.array_equal(grad_field._concentration_map, self.concentration_map)
        assert grad_field._step_size == 1
        assert grad_field._use_log is False
        assert grad_field._smooth is True

    def test_smoothing(self):
        smoothed_field = GradientField.smooth_field(self.concentration_map, self.conv_window)
        assert np.isclose(smoothed_field, self.mean_field).all()
        smoothed_field = GradientField.smooth_field(self.concentration_map, self.conv_window, stride=2)
        assert np.isclose(smoothed_field, self.mean_field[::2, ::2]).all()

    def test_log(self):
        smoothed_field = GradientField.smooth_field(self.concentration_map, self.conv_window)
        log_map = GradientField.compute_log(smoothed_field)
        assert np.isclose(log_map, np.log(self.mean_field)).all()

    def test_gradient(self):
        exp2d = lambda x, y : np.exp(-0.5 * (x**2 + y**2))

        space = np.linspace(-1, 1, 200)
        X, Y = np.meshgrid(space, space, indexing='ij')
        field = exp2d(X,Y)

        df = GradientField.compute_gradient(field, step=0.01)
        true_grad = np.stack([field * -X, field * -Y])
        # Ensure the derivatives match up to the step_size 0.01
        assert np.isclose(df, true_grad, atol=1e-2).all()

    def test_gradient_map(self):
        grad_field = GradientField(self.concentration_map, self.conv_window, step_size=1, use_log=False)
        true_grad = GradientField.compute_gradient(self.mean_field, step=1)
        true_norm = np.linalg.norm(true_grad, axis=0)

        gradient = grad_field.values
        assert gradient.shape == (2, 3, 3)
        assert np.isclose(gradient, true_grad).all()

        norm = grad_field.norm
        assert norm.shape == (3, 3)
        assert np.isclose(norm, true_norm).all()

    def test_gradient_interpolation(self):
        row_constant_field = np.repeat([np.arange(10)], repeats=10, axis=0)

        space = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(space, space, indexing='ij')

        conv_window = ConvolutionWindow.gaussian(3)
        grad_field = GradientField(row_constant_field, conv_window, use_log=False, smooth=False)
        assert np.array_equal(grad_field((0.5, 0.5)), grad_field((8.5, 0.5)))
        assert np.array_equal(grad_field((4.5, 4.5)), grad_field((8.5, 4.5)))
        dx, dy = grad_field((X,Y))
        assert np.array_equal(dx, np.zeros((100, 100)))
        assert np.isclose(dy, np.ones((100, 100))).all()

    def test_update_discret_space(self):
        grad_field = GradientField(self.concentration_map, self.conv_window, step_size=2)
        assert np.array_equal(grad_field.discret_space, 2 * [np.linspace(0.5, 4.5, 5),])
        grad_field._update_discret_space()
        assert np.array_equal(grad_field.discret_space, 2 * [[1.5, 3.5], ])
        # Take an other window and an other stride
        grad_field._conv_window = ConvolutionWindow.mean(2)
        grad_field._step_size = 3
        grad_field._update_discret_space()
        assert np.array_equal(grad_field.discret_space, 2 * [[1., 4.], ])
        grad_field._step_size = 2
        grad_field._update_discret_space()
        assert np.array_equal(grad_field.discret_space, 2 * [[1., 3.], ])
        # all ([np.array_equal(np.array(1.5, 3.5), discret_space) for discret_space in grad_field.discret_space])
