##
# @author Robin CREMESE <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2023-01-12 2:04:05 pm
 # @copyright https://mit-license.org/
 #
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from snake_ai.physim.convolution_window import ConvolutionWindow
from snake_ai.physim.regular_grid import RegularGrid2D
from snake_ai.utils.types import ArrayLike

from typing import List, Optional, Union
import logging


class SmoothGradientField:
    def __init__(self, concentration_map : ArrayLike, grid_2d : RegularGrid2D, conv_size : int, stride : int = 1) -> None:
        if not isinstance(concentration_map, (jax.Array, np.ndarray)):
            raise TypeError(f"Expected jax or numpy array instance for concentration map, get {type(concentration_map)} instead.")
        self._concentration_map = concentration_map

        if not isinstance(grid_2d, RegularGrid2D):
            raise TypeError(f"Expected RegularGrid2D instance for grid_2d, get {type(grid_2d)} instead.")
        self._grid_2d = grid_2d

        if not isinstance(stride, int) or stride <= 0:
            raise ValueError(f"Stride should be a positive integer. Get {stride} instead.")
        self._stride = stride

        self._dx, self._dy = None, None
        self._interp_dx, self._interp_dy = None, None

    @property
    def dx(self):
        "Gradient field along x axis"
        return self._dx

    @property
    def dy(self):
        "Gradient field along y axis"
        return self._dy

    def get_gauss_diff_window(self, size):
        gaussian = ConvolutionWindow.gaussian(size)
        pass

    def __call__(self, x, y, method : str = 'linear') -> jax.Array:
        pass

    def __repr__(self) -> str:
        return f"{__class__.__name__}() "
class GradientField:
    def __init__(self, concentration_map : ArrayLike, conv_window : Optional[ConvolutionWindow] = None, step_size : int = 1,
                 use_log : bool = False) -> None:
        if not isinstance(concentration_map, (jax.Array, np.ndarray)):
            raise TypeError(f"Expected jax or numpy array instance, get {type(concentration_map)} instead.")
        assert concentration_map.ndim in range(2,4), "Accepted number of dims : 2, 3"
        self._concentration_map = concentration_map

        if (conv_window is not None) and (not isinstance(conv_window, ConvolutionWindow)):
            raise TypeError(f"Convolution window should be None or an instance of ConvolutionWindow, not {type(conv_window)}")
        self._conv_window = conv_window

        if step_size <= 0:
            raise ValueError(f"Step should be a positive value, not {step_size}")
        self._step_size = step_size

        self._use_log = use_log
        self._smooth = conv_window is not None
        # Concentration maps are supposed to start at 0.5 and are evenly spaced with step size = 1
        self.discret_space = [np.linspace(0.5, dim - 0.5, dim) for dim in self._concentration_map.shape]
        # Fields to be defined in private methods
        self._gradient_map : Optional[jax.Array] = None
        self._interlators : Optional[List[jsp.interpolate.RegularGridInterpolator]] = None

    ## Protected methods
    def _compute_gradient_map(self):
        conc_field = self._concentration_map
        if self._smooth:
            conc_field = self.smooth_field(conc_field, self._conv_window, self._step_size)
            self._update_discret_space()
        if self._use_log:
            conc_field = self.compute_log(conc_field)
        self._gradient_map = self.compute_gradient(conc_field, self._step_size)

    def _init_interpolators(self):
        self._interlators = [jsp.interpolate.RegularGridInterpolator(self.discret_space, grad, fill_value=None) for grad in self.values]

    def _update_discret_space(self):
        window_size = self._conv_window.size
        stride = self._step_size

        # self.discret_space = [np.linspace(window_size / 2, dim - window_size / 2, (dim - window_size) // stride + 1) for dim in self._concentration_map.shape]
        self.discret_space = [np.arange(window_size / 2, dim + 1 - window_size / 2, stride) for dim in self._concentration_map.shape]

    ## Properties
    @property
    def values(self) -> jax.Array:
        "Values of the gradient field"
        if self._gradient_map is None:
            self._compute_gradient_map()
        return self._gradient_map

    @property
    def norm(self) -> jax.Array:
        "Norm of the gradient field in each point of space"
        return jnp.linalg.norm(self.values, axis=0)

    ## Static methods
    @staticmethod
    def smooth_field(field : ArrayLike, conv_window : ConvolutionWindow, stride = 1) -> jax.Array:
        if not isinstance(conv_window, ConvolutionWindow):
            raise TypeError(f"Expected ConvolutionWindow instance, get {type(conv_window)} instead.")

        smoothed_field = jsp.signal.convolve(field, conv_window.value, mode='same')

        return smoothed_field[::stride, ::stride]

    @staticmethod
    def compute_log(field : ArrayLike, eps : float = 1e-6) -> jax.Array:
        return jnp.log(jnp.where(field < eps, eps, field))

    @staticmethod
    def compute_gradient(field : ArrayLike, step : Union[float, List[ArrayLike]] = 1.):
        assert field.ndim == 2, "Computation allowed only for field of dim 2"
        # Constant step size for gradient estimation
        # if isinstance(step, (float, int)):
        assert step > 0, f"Only constant positive numbers are allowed for step sizes, not {step}"
        return jnp.stack(jnp.gradient(field, step))
        # Value positions in each of the dimensions
        # if isinstance(step, list):
        #     assert [len(step_array) == field_dim for step_array, field_dim in zip(step, field.shape)]
        #     return jnp.stack(jnp.gradient(field, *step))

    # TODO : Implement factory methods to initialise the gradient field + make conv_layer optional

    ## Dunder methods
    def __call__(self, positions : ArrayLike, method : str = 'linear') -> jax.Array:
        assert method in ['linear', 'nearest', 'cubic'], f"Accepted interpolation methods are 'linear', 'nearest', 'cubic', not {method}"

        if self._interlators is None:
            self._init_interpolators()
        return jnp.stack([interpolator(positions, method) for interpolator in self._interlators]).squeeze()

    def __repr__(self) -> str:
        return f"{__class__.__name__}(concentration_map.shape={self._concentration_map.shape!r}, conv_window={self._conv_window!r}, step={self._step_size}, use_log={self._use_log})"