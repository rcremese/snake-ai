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
import warnings

def smooth_field(field : ArrayLike, conv_window : ConvolutionWindow, stride = 1, mode='same') -> jax.Array:
    assert mode.lower() in ['same', 'valid'], f"Accepted convolution modes, are 'same' and 'valid' (see https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.signal.convolve.html)"
    if not isinstance(conv_window, ConvolutionWindow):
        raise TypeError(f"Expected ConvolutionWindow instance, get {type(conv_window)} instead.")

    smoothed_field = jsp.signal.convolve(field, conv_window.value, mode=mode.lower())

    return smoothed_field[::stride, ::stride]

def compute_log(field : ArrayLike, eps : float = 1e-6) -> jax.Array:
    return jnp.log(jnp.where(field < eps, eps, field))

def compute_gradient(field : ArrayLike, step : Union[float, List[ArrayLike]] = 1.):
    assert field.ndim == 2, "Computation allowed only for field of dim 2"
    # Constant step size for gradient estimation
    # if isinstance(step, (float, int)):
    assert step > 0, f"Only constant positive numbers are allowed for step sizes, not {step}"
    return jnp.stack(jnp.gradient(field, step))

class SmoothGradientField:
    def __init__(self, concentration_map : ArrayLike, grid_2d : RegularGrid2D, conv_size : int, stride : int = 1, conv_mode : str = 'same') -> None:
        """Class that represent a 2D smoothed gradient field of the input concentration map.

        Differentiation and smoothing are done by a convolution with 2D differentiated gaussian filters : \delta_x G(x,y) and \delta_y G(x,y)

        Args:
            concentration_map (ArrayLike): 2D array representing the concentration map. The first axis corresponds to the x dim and the second axis to the y dim.
            grid_2d (RegularGrid2D): object that represent the space coordinates of the points in the concentration map
            conv_size (int): size of the convolution windows to use in terms of the concentration map spacing
            stride (int, optional): stride used for the convolution. Also used to downgrade the resolution of 2D grid. Defaults to 1.
            conv_mode (str, optional): method that specify the padding used when applying convolution. Accepted values are 'full' and 'valid'. Defaults to 'full'.
        Raises:
            TypeError: if the inputs are not of the expected types
            ValueError: if the concentration map is not a 2D array or the stride and conv_size are < 1
        """
        if not isinstance(concentration_map, (jax.Array, np.ndarray)):
            raise TypeError(f"Expected jax or numpy array instance for concentration map, get {type(concentration_map)} instead.")
        if concentration_map.ndim != 2:
            raise ValueError(f"Smooth gradient field estimations can only be done on 2D maps, not {concentration_map.ndim} dim.")
        self._concentration_map = concentration_map

        if not isinstance(grid_2d, RegularGrid2D):
            raise TypeError(f"Expected RegularGrid2D instance for grid_2d, get {type(grid_2d)} instead.")
        self._grid_2d = grid_2d

        if not isinstance(stride, int) or stride <= 0:
            raise ValueError(f"Stride should be a positive integer. Get {stride} instead.")
        self._stride = stride

        # Convolution windows instanciation
        if conv_size < 1 :
            raise ValueError(f"Convolution window size should be at least 1 pixel. Get {conv_size} instead.")
        self._conv_dx = ConvolutionWindow.gaussian_dx(conv_size)
        self._conv_dy = ConvolutionWindow.gaussian_dy(conv_size)

        if not conv_mode.lower() in ['valid', 'same']:
            raise ValueError(f"Unknown convolution padding method {conv_mode}. Expected 'valid' or 'full'.")
        self._conv_mode = conv_mode.lower()
        # Attributs that will be defined in proper methods
        self._dx, self._dy = None, None
        self._interp_dx, self._interp_dy = None, None

    @property
    def dx(self):
        "Gradient field along x axis"
        if self._dx is None:
            self._dx = smooth_field(self._concentration_map, self._conv_dx, self._stride, mode=self._conv_mode)
            self._grid_2d.convolution_dowgrade(self._conv_dx, self._stride, mode=self._conv_mode, axis='x')
        return self._dx

    @property
    def dy(self):
        "Gradient field along y axis"
        if self._dy is None:
            self._dy = smooth_field(self._concentration_map, self._conv_dy, self._stride, mode=self._conv_mode)
            self._grid_2d.convolution_dowgrade(self._conv_dy, self._stride, mode=self._conv_mode, axis='y')
        return self._dy

    def _check_extrapolation(self, x : Union[float, ArrayLike], y : Union[float, ArrayLike]):
        if isinstance(x, float) and (x < self._grid_2d.x_init or x > self._grid_2d.x_end):
            warnings.warn( f"Position {x} is out of the domain [{self._grid_2d.x_init}, {self._grid_2d.x_end}] and then field value is extrapolated !")
        if isinstance(y, float) and (y < self._grid_2d.y_init or y > self._grid_2d.y_end):
            warnings.warn( f"Position {y} is out of the domain [{self._grid_2d.y_init}, {self._grid_2d.y_end}] and then field value is extrapolated !")
        if isinstance(x, (np.ndarray, jax.Array)) and ((x < self._grid_2d.x_init).any() or (x > self._grid_2d.x_end).any()):
            extra_pos, _ = np.where(x < self._grid_2d.x_init | x > self._grid_2d.x_end) # get the index of the extrapolated values
            warnings.warn( f"Vector x is out of the domain [{self._grid_2d.x_init}, {self._grid_2d.x_end}] at positions {extra_pos}. Field values at this positions are extrapolated !" )
        if isinstance(y, (np.ndarray, jax.Array)) and ((y < self._grid_2d.y_init).any() or (y > self._grid_2d.y_end).any()):
            extra_pos, _ = np.where(y < self._grid_2d.y_init | y > self._grid_2d.y_end) # get the index of the extrapolated values
            warnings.warn( f"Vector y is out of the domain [{self._grid_2d.y_init}, {self._grid_2d.y_end}] at positions {extra_pos}. Field values at this positions are extrapolated !" )

    def __call__(self, x : Union[float, ArrayLike], y : Union[float, ArrayLike], method : str = 'linear') -> jax.Array:
        assert method.lower() in ["linear", "nearest", "cubic"], f"Unkown method {method}. Accepted methods are 'linear', 'nearest' and 'cubic'. \
            See scipy.interpolate.RegularGridInterpolator for further detail"
        # Instanciation of the interpolators if not defined
        if self._interp_dx is None:
            self._interp_dx = jsp.interpolate.RegularGridInterpolator((self._grid_2d.x, self._grid_2d.y), self.dx, fill_value=None) # extrapolation is allowed
        if self._interp_dy is None:
            self._interp_dy = jsp.interpolate.RegularGridInterpolator((self._grid_2d.x, self._grid_2d.y), self.dy, fill_value=None)
        # Check if desired values are out of the domain
        self._check_extrapolation(x, y)
        return self._interp_dx((x, y), method=method.lower()), self._interp_dy((x, y), method = method.lower())

    def __repr__(self) -> str:
        return f"{__class__.__name__}(concentration_map={self._concentration_map}, grid_2d={self._grid_2d!r}, conv_size={self._conv_size}, stride={self._stride}) "
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