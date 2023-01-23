
import numpy as np
from scipy import signal
import jax

SIGMA_MAX = 3

def exp_2d(x, y, mu = 0, std = 1):
    return np.exp(-0.5 * ((x - mu)**2 + (y - mu)**2) / (std**2) )
class ConvolutionWindow:
    def __init__(self, size : int, conv_type : str, std : float = 1) -> None:
        if not conv_type.lower() in ['gaussian', 'mean', 'gaussian_dx', 'gaussian_dy']:
            raise ValueError(f"Unknown type {conv_type}. Expected 'gaussian', 'diff_gaussian' or 'mean'.")
        self.conv_type = conv_type.lower()

        if size <= 0:
            raise ValueError(f"Convolution window can not be negative.")
        self.size = int(size)

        if std <= 0:
            raise ValueError(f"Scale parameter need to be positive. Get {std}")
        self.std = std
        # Set to None the priate variables
        self._conv_window = None

    def _compute_convolution_window(self) -> jax.Array:
        if self.conv_type == 'mean':
            return np.ones((self.size, self.size)) / self.size**2

        # gaussian filters
        space = np.linspace(- 0.5 * self.size, 0.5 * self.size, self.size)
        X, Y = np.meshgrid(space, space, indexing='xy')
        gaussian_2d = exp_2d(X, Y, std=self.std)
        if self.conv_type == 'gaussian':
            return gaussian_2d

        # differential cases
        var = self.std ** 2
        if self.conv_type == 'gaussian_dx':
            return - X * gaussian_2d / var
        if self.conv_type == 'gaussian_dy':
            return - Y * gaussian_2d / var
        # return jsp.stats.norm.pdf(x[:, jnp.newaxis], loc=shift, scale=self.scale) * jsp.stats.norm.pdf(x, loc=shift, scale=self.scale)

        # Unknown conv_type
        raise ValueError(f"Unknown conv_type {self.conv_type}")

    @classmethod
    def mean(cls, size):
        return cls(size, 'mean')

    @classmethod
    def gaussian(cls, size, std = None):
        # set the scale for the window to be centered and with border at +/- 3\sigma
        if std is None:
            std = size / (2 * SIGMA_MAX)
        return cls(size, 'gaussian', std)

    @classmethod
    def gaussian_dx(cls, size, std = None):
        if std is None:
            std = size / (2 * SIGMA_MAX)
        return cls(size, 'gaussian_dx', std)

    @classmethod
    def gaussian_dy(cls, size, std = None):
        if std is None:
            std = size / (2 * SIGMA_MAX)
        return cls(size, 'gaussian_dy', std)

    @property
    def value(self) -> jax.Array:
        "Convolution window"
        if self._conv_window is None:
            self._conv_window = self._compute_convolution_window()
        return self._conv_window

    def __repr__(self) -> str:
        return f"{__class__.__name__}(size={self.size}, conv_type={self.conv_type}, std={self.std:1.3f})"