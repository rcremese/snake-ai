import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from snake_ai.physim import ConvolutionWindow
from typing import Union

ArrayLike = Union[np.ndarray, jax.Array]

class GradientField:
    # def __init__(self, concentration_map : ArrayLike, use_log : bool = True) -> None:
    #     if not isinstance(concentration_map, (jax.Array, np.ndarray)):
    #         raise TypeError(f"Expected jax or numpy array instance, get {type(concentration_map)} instead.")
    #     assert concentration_map.ndim == 2, "Accepted number of dims : 2, 3"
    #     self._concentration_map = concentration_map
    #     self.use_log = use_log

    #     self._gradient_map = None

    @staticmethod
    def smooth_field(field : ArrayLike, conv_window : ConvolutionWindow, stride = 0) -> jax.Array:
        if not isinstance(conv_window, ConvolutionWindow):
            raise TypeError(f"Expected ConvolutionWindow instance, get {type(ConvolutionWindow)} instead.")

        if stride == 0:
            return jsp.signal.convolve(field, conv_window.value, mode='same')

        raise NotImplementedError("Stride is not yet implemented")

    @staticmethod
    def compute_log(field : ArrayLike, eps : float = 1e-6) -> jax.Array:
        return jnp.log(jnp.where(field < eps, eps, field))

    @staticmethod
    def compute_gradient(field : ArrayLike, step_size : float = 1.):
        assert field.ndim == 2, "Computation allowed only for field of dim 2"
        assert isinstance(step_size, float) and (step_size > 0), f"Only constant positive float numbers are allowed for step sizes, not {step_size}"
        dy, dx = jnp.gradient(field, step_size)
        return jnp.stack([dx, dy])