
import jax.numpy as jnp
import jax.scipy as jsp
import jax

class ConvolutionWindow:
    def __init__(self, size : int, type : str, scale = 1) -> None:
        if not type.lower() in ['gaussian', 'mean']:
            raise TypeError(f"Unknown type {type}. Expected 'gaussian' or 'mean'.") 
        self.type = type.lower()

        if size <= 0:
            raise ValueError(f"Convolution window can not be negative.") 
        self.size = int(size) 

        if self.type == 'mean':
            self._conv_window = jnp.ones((size, size)) / size**2
        elif self.type == 'gaussian':
            x = jnp.arange(size)
            shift = self.size / 2
            # Trick to get a gaussian convolutional array (from https://jax.readthedocs.io/en/latest/notebooks/convolutions.html?highlight=convolution#basic-n-dimensional-convolution)
            self._conv_window = jsp.stats.norm.pdf(x[:, jnp.newaxis], loc=shift, scale=scale) * \
                jsp.stats.norm.pdf(x, loc=shift, scale=scale)
    
    @classmethod
    def gaussian(cls, size, scale=1):
        return cls(size, 'gaussian', scale).window

    @classmethod
    def mean(cls, size):
        return cls(size, 'mean').window

    @property
    def window(self) -> jax.Array:
        "Convolution window"
        return self._conv_window