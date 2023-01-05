
import jax.numpy as jnp
import jax.scipy as jsp
import jax

class ConvolutionWindow:
    def __init__(self, size : int, conv_type : str, scale = 1) -> None:
        if not conv_type.lower() in ['gaussian', 'mean']:
            raise TypeError(f"Unknown type {conv_type}. Expected 'gaussian' or 'mean'.")
        self.conv_type = conv_type.lower()

        if size <= 0:
            raise ValueError(f"Convolution window can not be negative.")
        self.size = int(size)

        if scale <= 0:
            raise ValueError(f"Scale parameter need to be positive. Get {scale}")
        self.scale = scale
        # Set to None the priate variables
        self._conv_window = None

    def _compute_convolution_window(self) -> jax.Array:
        if self.conv_type == 'mean':
            return jnp.ones((self.size, self.size)) / self.size**2
        if self.conv_type == 'gaussian':
            x = jnp.arange(self.size)
            shift = self.size / 2
            # Trick to get a gaussian convolutional array (from https://jax.readthedocs.io/en/latest/notebooks/convolutions.html?highlight=convolution#basic-n-dimensional-convolution)
            return jsp.stats.norm.pdf(x[:, jnp.newaxis], loc=shift, scale=self.scale) * jsp.stats.norm.pdf(x, loc=shift, scale=self.scale)

    @classmethod
    def gaussian(cls, size, scale=1):
        return cls(size, 'gaussian', scale)

    @classmethod
    def mean(cls, size):
        return cls(size, 'mean')

    @property
    def value(self) -> jax.Array:
        "Convolution window"
        if self._conv_window is None:
            self._conv_window = self._compute_convolution_window()
        return self._conv_window

    def __repr__(self) -> str:
        return f"{__class__.__name__}(size={self.size}, conv_type={self.conv_type}, scale={self.scale})"