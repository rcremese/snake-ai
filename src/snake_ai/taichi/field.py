import taichi as ti
import taichi.math as tm
import numpy as np

from snake_ai.envs.geometry import Rectangle, Cube
from snake_ai.taichi.boxes import Box2D, Box3D, convert_cube, convert_rectangle
from snake_ai.taichi.maths import lerp

from typing import Union
from enum import Enum
from abc import ABC, abstractmethod


@ti.func
def map_value_to_idx(value: float, x_min: float, x_max: float, step_size: float) -> int:
    """Map a physical value to the index of the corresponding cell in the field.

    Args:
        value (float): value to map
        x_min (float): minimum value of the field
        x_max (float): maximum value of the field
        step_size (float): step size of the field

    Returns:
        int: The index of the lower cell in the field that contains the value.
        If the value is under the lower bound of the field, return -1.
        If the value is greater than the upper bound of the field, return -1.
    """
    assert step_size > 0, "Expected step_size to be positive. Get {}".format(step_size)
    if value < x_min:
        idx = -1
    elif value > x_max:
        idx = -2
    else:
        idx = int((value - x_min) / step_size)
    return idx


@ti.data_oriented
class SampledField(ABC):
    _values: ti.Field
    _bounds: Union[Rectangle, Cube]
    step_sizes: list[float]
    dim: int

    @abstractmethod
    def __init__(
        self,
        values: np.ndarray,
        bounds: Union[Rectangle, Cube],
        needs_grad: bool = False,
    ) -> None:
        raise NotImplementedError

    @ti.func
    def at(self, pos: ti.template()) -> ti.template():
        if self.dim == 2:
            return self._at_2d(pos)
        elif self.dim == 3:
            return self._at_3d(pos)
        else:
            raise NotImplementedError

    @ti.func
    def _at_2d(self, pos: tm.vec2) -> ti.template():
        idx = tm.ivec2([0, 0])
        toi = tm.vec2([0, 0])
        for i in ti.static(range(2)):
            temp_idx = map_value_to_idx(
                pos[i], self._bounds.min[i], self._bounds.max[i], self.step_sizes[i]
            )
            # Handle extrapolation
            if temp_idx == -1:
                idx[i] = 0
                toi[i] = 0.0
            elif temp_idx == -2:
                idx[i] = self._values.shape[i] - 2
                toi[i] = 1.0
            else:
                idx[i] = temp_idx
                toi[i] = pos[i] - self._bounds.min[i] - idx[i] * self.step_sizes[i]
        # Interpolation along x-axis
        s0 = lerp(self._values[idx], self._values[idx + tm.ivec2([1, 0])], toi[0])
        s1 = lerp(
            self._values[idx + tm.ivec2([0, 1])],
            self._values[idx + tm.ivec2([1, 1])],
            toi[0],
        )
        # Interpolation along y-axis
        return lerp(s0, s1, toi[1])

    @ti.func
    def _at_3d(self, pos: tm.vec3) -> ti.template():
        idx = tm.ivec3([0, 0, 0])
        toi = tm.vec3([0, 0, 0])
        for i in ti.static(range(3)):
            temp_idx = map_value_to_idx(
                pos[i], self._bounds.min[i], self._bounds.max[i], self.step_sizes[i]
            )
            # Handle extrapolation
            if temp_idx == -1:
                idx[i] = 0
                toi[i] = 0.0
            elif temp_idx == -2:
                idx[i] = self._values.shape[i] - 2
                toi[i] = 1.0
            else:
                idx[i] = temp_idx
                toi[i] = pos[i] - self._bounds.min[i] - idx[i] * self.step_sizes[i]
        # Interpolation along x-axis
        s0 = lerp(
            self._values[idx],
            self._values[idx + tm.ivec3([1, 0, 0])],
            toi[0],
        )
        s1 = lerp(
            self._values[idx + tm.ivec3([0, 1, 0])],
            self._values[idx + tm.ivec3([1, 1, 0])],
            toi[0],
        )
        s2 = lerp(
            self._values[idx + tm.ivec3([0, 0, 1])],
            self._values[idx + tm.ivec3([1, 0, 1])],
            toi[0],
        )
        s3 = lerp(
            self._values[idx + tm.ivec3([0, 1, 1])],
            self._values[idx + tm.ivec3([1, 1, 1])],
            toi[0],
        )
        # Interpolation along y-axis
        s4 = lerp(s0, s1, toi[1])
        s5 = lerp(s2, s3, toi[1])
        # Interpolation along z-axis
        return lerp(s4, s5, toi[2])

    @ti.func
    def contains(self, pos: ti.template()) -> bool:
        return self._bounds.contains(pos)

    def __getitem__(self, idx: ti.template()) -> float:
        return self._values[idx]

    @property
    def resolution(self):
        return self._values.shape

    @property
    def linspace(self) -> list[np.ndarray]:
        return [
            np.linspace(
                self._bounds.min[i],
                self._bounds.max[i],
                self._values.shape[i],
            )
            for i in range(self.dim)
        ]

    @property
    def meshgrid(self) -> np.ndarray:
        return np.meshgrid(*self.linspace, indexing="ij")

    @property
    def values(self) -> np.ndarray:
        return self._values.to_numpy()

    @property
    def bounds(self) -> Union[Rectangle, Cube]:
        if self.dim == 2:
            return Rectangle(
                self._bounds.min[0],
                self._bounds.min[1],
                self._bounds.width,
                self._bounds.height,
            )
        elif self.dim == 3:
            return Cube(
                self._bounds.min[0],
                self._bounds.min[1],
                self._bounds.min[2],
                self._bounds.width,
                self._bounds.height,
                self._bounds.depth,
            )
        else:
            raise ValueError("Only 2D and 3D fields are supported")


# TODO : Take into account extrapolation
@ti.data_oriented
class ScalarField(SampledField):
    def __init__(
        self,
        values: np.ndarray,
        bounds: Union[Rectangle, Cube],
        needs_grad: bool = False,
    ) -> None:
        ## Dimension and values
        assert (
            values.ndim == 2 or values.ndim == 3
        ), f"Expected sampled field to be 2D or 3D. Get {values.ndim}D"
        self._values = ti.field(dtype=ti.f32, shape=values.shape, needs_grad=needs_grad)
        self._values.from_numpy(values)
        self.dim = values.ndim
        ## Bounds and step sizes
        if isinstance(bounds, Rectangle):
            assert self.dim == 2, f"Expected bounds to be {self.dim}D. Get 2D bounds"
            self._bounds = convert_rectangle(bounds)
        elif isinstance(bounds, Cube):
            assert self.dim == 3, f"Expected bounds to be {self.dim}D. Get 3D bounds"
            self._bounds = convert_cube(bounds)
        else:
            ## Case where bounds is a Box2D or Box3D
            self._bounds = bounds
            # raise TypeError("Expected bounds to be an instance of Rectangle or Cube")

        self.step_sizes = [
            (self._bounds.max[i] - self._bounds.min[i]) / (self._values.shape[i] - 1)
            for i in range(self.dim)
        ]

    def __repr__(self) -> str:
        return f"ScalarField(values={self._values.shape}, bounds={self._bounds}, extrapolation={self._extrapolation})"


@ti.data_oriented
class VectorField(SampledField):
    def __init__(
        self,
        values: np.ndarray,
        bounds: Union[Box2D, Box3D],
        needs_grad: bool = False,
    ) -> None:
        ## Dimension and values
        dim = values.shape[0]
        assert (dim == 2 and values.ndim == 3) or (
            dim == 3 and values.ndim == 4
        ), f"Expected sampled field to be 2D or 3D vector fields. Get {dim}D"
        self.dim = dim
        self._values = ti.Vector.field(
            dim, dtype=ti.f32, shape=values.shape[1:], needs_grad=needs_grad
        )
        self._values.from_numpy(np.moveaxis(values, 0, -1))

        ## Bounds and step sizes
        if isinstance(bounds, Rectangle):
            assert self.dim == 2, f"Expected bounds to be {self.dim}D. Get 2D bounds"
            self._bounds = convert_rectangle(bounds)
        elif isinstance(bounds, Cube):
            assert self.dim == 3, f"Expected bounds to be {self.dim}D. Get 3D bounds"
            self._bounds = convert_cube(bounds)
        else:
            ## Case where bounds is a Box2D or Box3D
            self._bounds = bounds
            # raise TypeError("Expected bounds to be an instance of Rectangle or Cube")

        self.step_sizes = [
            (self._bounds.max[i] - self._bounds.min[i]) / (self._values.shape[i] - 1)
            for i in range(self.dim)
        ]

    @property
    @ti.kernel
    def max(self) -> float:
        max_val = -tm.inf
        for I in ti.grouped(self._values):
            max_val = ti.max(max_val, tm.length(self._values[I]))
        return max_val

    def __repr__(self) -> str:
        return f"VectorField(values={self._values.shape}, bounds={self._bounds}, extrapolation={self._extrapolation})"


@ti.kernel
def clip(vector_field: VectorField, value: float):
    for I in ti.grouped(vector_field._values):
        if tm.length(vector_field._values[I]) > value:
            vector_field._values[I] = (
                vector_field._values[I] / tm.length(vector_field._values[I]) * value
            )


def spatial_gradient(
    scalar_field: ScalarField, needs_grad: bool = False
) -> VectorField:
    """Compute the spatial gradient of a scalar field.

    Args:
        scalar_field (ScalarField): _description_

    Returns:
        VectorField: _description_
    """
    assert isinstance(
        scalar_field, ScalarField
    ), f"Expected a ScalarField, get {type(scalar_field)})"
    values = scalar_field._values.to_numpy()

    grad_values = np.stack(
        np.gradient(values, *scalar_field.step_size, edge_order=2),
        dtype=np.float32,
    )
    return VectorField(
        grad_values,
        scalar_field._bounds,
        scalar_field._extrapolation,
        needs_grad,
    )


def log(
    scalar_field: ScalarField, needs_grad: bool = False, eps: float = 1e-5
) -> ScalarField:
    """Compute the spatial gradient of a scalar field.

    Args:
        scalar_field (ScalarField): _description_

    Returns:
        VectorField: _description_
    """
    assert isinstance(
        scalar_field, ScalarField
    ), f"Expected a ScalarField, get {type(scalar_field)})"
    values = scalar_field._values.to_numpy()
    log_values = np.log(np.where(values > eps, values, eps))
    return ScalarField(
        log_values,
        scalar_field._bounds,
        needs_grad,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ti.init(arch=ti.cpu)

    space = np.linspace(-1, 1, 100)
    values = space[:, None] * space[None, :]
    vector = np.stack(np.gradient(values), dtype=np.float32)

    bounds = Box2D(tm.vec2([-1, -1]), tm.vec2([1, 1]))
    field = ScalarField(values, bounds)
    vector_field = spatial_gradient(field)
    # VectorField(vector, bounds)

    print("res", field.resolution, "step", field.step_size)

    print(
        field.at(tm.vec2([0, 0])),
        field.at(tm.vec2([1, 1])),
        field.at(tm.vec2([1, 0])),
        field.at(tm.vec2([0, 1])),
    )
    print(vector_field.at(tm.vec2([0, 0])), vector[:, 50, 50], vector[:, 49, 49])
    print(vector_field.max, np.max(np.linalg.norm(vector_field._values, axis=-1)))

    plt.imshow(values, origin="lower")
    plt.quiver(vector_field._values[:, :, 0], vector_field._values[:, :, 1], color="k")
    plt.show()
