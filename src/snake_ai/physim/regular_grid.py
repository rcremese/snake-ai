from typing import Optional, List
from snake_ai.utils.types import Numerical
from snake_ai.physim.convolution_window import ConvolutionWindow
import numpy as np

class RegularGrid2D:
    def __init__(self, x_init : Numerical, x_end : Numerical, x_step : Numerical = 1, y_init : Optional[Numerical] = None, y_end : Optional[Numerical] = None, y_step : Optional[Numerical] = None):
        ## Regular grid along x axis
        self._check_type(x_init, 'x_init')
        self.x_init = x_init

        self._check_type(x_end, 'x_end')
        self.x_end = x_end

        self._check_type(x_step, 'x_step')
        self.x_step = x_step

        ## Regular grid along y axis
        self.y_init = x_init if (y_init is None) else y_init
        self._check_type(self.y_init, 'y_init')
        self.y_end = x_end if (y_end is None) else y_end
        self._check_type(self.y_end, 'y_end')
        self.y_step = x_step if (y_end is None) else y_step
        self._check_type(self.y_step, 'y_step')

        self.x = np.arange(self.x_init, self.x_end, self.x_step)
        self.y = np.arange(self.y_init, self.y_end, self.y_step)

    def convolution_dowgrade(self, conv_window : ConvolutionWindow, stride : int, mode : str = 'same', axis : str = 'both'):
        assert isinstance(stride, int) and stride > 0, f"Expected stride should be an int > 0, not {stride}"
        assert mode.lower() in ['same', 'valid'], f"Accepted modes are 'same' and 'valid', not {mode}"
        assert axis.lower() in ['x', 'y', 'both'], f"Accepted axis are 'x', 'y' and 'both', not {axis}"
        # Starting point differs depending on the selected mode
        if mode.lower() == 'same':
            start = 0
        else:
            start = conv_window.size // 2

        assert stride + start < len(self.x) and stride + start < len(self.y), \
            f"Stride + starting position should be lower than the grid dimensions ({len(self.x)}, {len(self.y)}). Get {stride + start}."

        if axis.lower() in ['x', 'both']:
            self.x = self.x[start::stride]
            self.x_init, self.x_end, self.x_step = self.x[0], self.x[-1], self.x[1] - self.x[0]
        if axis.lower() in ['y', 'both']:
            self.y = self.y[start::stride]
            self.y_init, self.y_end, self.y_step = self.y[0], self.y[-1], self.y[1] - self.y[0]

    @classmethod
    def regular_square(cls, *args : List[Numerical]):
        if len(args) == 1:
            assert args[0] > 1, "The end value need to be > 1 as the class is an implementation of np.arange(0, end, 1)"
            start, end, step = 0, args[0], 1
        elif len(args) == 2:
            start, end, step = args[0], args[1], 1
        elif len(args) == 3:
            start, end, step = args[0], args[1], args[2]
        else :
            raise LookupError(f"Only 3 arguments are accepted for regular_square(). Got {len(args)}")
        return cls(start, end, step)

    @classmethod
    def unitary_rectangle(cls, x_end : Numerical, y_end : Numerical):
        return cls(0, x_end, 1, 0, y_end, 1)

    @property
    def mesh(self) -> np.ndarray:
        "Meshgrid of the space"
        return np.meshgrid(self.x, self.y, indexing='xy')

    def _check_type(self, val : Numerical, name : str):
        if not isinstance(val, (float, int)):
            raise TypeError(f"Expected int or float for {name}, not {type(val)}")

    def __repr__(self) -> str:
        return f"{__class__.__name__}(x_init={self.x_init}, x_end={self.x_end}, x_step={self.x_step}, y_init={self.y_init}, y_end={self.y_end}, y_step={self.y_step})"
