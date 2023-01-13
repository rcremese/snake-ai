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

from snake_ai.physim import ConvolutionWindow
from snake_ai.utils.types import ArrayLike

from typing import List, Optional

class GradientField:
    def __init__(self, concentration_map : ArrayLike, conv_window : ConvolutionWindow, step : int = 1, use_log : bool = True) -> None:
        if not isinstance(concentration_map, (jax.Array, np.ndarray)):
            raise TypeError(f"Expected jax or numpy array instance, get {type(concentration_map)} instead.")
        assert concentration_map.ndim in range(2,4), "Accepted number of dims : 2, 3"
        self._concentration_map = concentration_map

        if not isinstance(conv_window, ConvolutionWindow):
            raise TypeError(f"Convolution window should be instance of ConvolutionWindow, not {type(conv_window)}")
        self._conv_window = conv_window

        self.use_log = use_log

        assert step > 0, f"Step should be a positive value, not {step}"
        self._step = step

        self._discret_space = [np.arange(0, dim, self._step) for dim in self._concentration_map.shape]
        # Fields to be defined in private methods
        self._gradient_map : Optional[jax.Array] = None
        self._interlators : Optional[List[jsp.interpolate.RegularGridInterpolator]] = None

    ## Protected methods
    def _compute_gradient_map(self):
        conc_field = self.smooth_field(self._concentration_map, self._conv_window, self._step)
        if self.use_log:
            conc_field = self.compute_log(conc_field)
        self._gradient_map = self.compute_gradient(conc_field, self._step)

    def _init_interpolators(self):
        self._interlators = [jsp.interpolate.RegularGridInterpolator(self._discret_space, grad) for grad in self.values]

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

        if stride == 1:
            return jsp.signal.convolve(field, conv_window.value, mode='same')

        raise NotImplementedError("Stride is not yet implemented")

    @staticmethod
    def compute_log(field : ArrayLike, eps : float = 1e-6) -> jax.Array:
        return jnp.log(jnp.where(field < eps, eps, field))

    @staticmethod
    def compute_gradient(field : ArrayLike, step_size : float = 1.):
        assert field.ndim == 2, "Computation allowed only for field of dim 2"
        assert step_size > 0, f"Only constant positive numbers are allowed for step sizes, not {step_size}"
        return jnp.stack(jnp.gradient(field, step_size))

    ## Dunder methods
    def __call__(self, positions : ArrayLike, method : str = 'linear') -> jax.Array:
        assert method in ['linear', 'nearest', 'cubic'], f"Accepted interpolation methods are 'linear', 'nearest', 'cubic', not {method}"

        if self._interlators is None:
            self._init_interpolators()
        return jnp.stack([interpolator(positions, method) for interpolator in self._interlators])

    def __repr__(self) -> str:
        return f"{__class__.__name__}(concentration_map.shape={self._concentration_map.shape!r}, conv_window={self._conv_window!r}, step={self._step}, use_log={self.use_log})"