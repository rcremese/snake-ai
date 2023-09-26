import taichi as ti
import taichi.math as tm
import numpy as np

from snake_ai.taichi.geometry import Box2D, Box3D
from snake_ai.taichi.maths import lerp

from typing import Union
from enum import Enum
from abc import ABC, abstractmethod


class Extrapolation(Enum):
    ZERO = 0
    SYMETRIC = 1
    PERIODIC = 2


@ti.func
def map_value_to_idx(value: float, x_min: float, x_max: float, step_size: float) -> int:
    """Map a physical value to the index of the corresponding cell in the field.

    Args:
        value (float): value to map
        x_min (float): minimum value of the field
        x_max (float): maximum value of the field
        step_size (float): step size of the field

    Returns:
        int: The index of the lower cell in the field that contains the value. If the value is outside the field, return -1.
    """
    assert step_size > 0, "Expected step_size to be positive. Get {}".format(step_size)
    clamped_value = tm.clamp(value, x_min, x_max)
    idx = int((clamped_value - x_min) / step_size)

    if clamped_value != value:
        idx = -1
    return idx


@ti.data_oriented
class SampledField(ABC):
    values: ti.Field
    bounds: Union[Box2D, Box3D]
    _step_sizes: list[float]

    @abstractmethod
    def __init__(
        self,
        values: np.ndarray,
        bounds: Union[Box2D, Box3D],
        extrapolation: Extrapolation = Extrapolation.ZERO,
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
            # idx[i] = map_value_to_idx(
            temp_idx = map_value_to_idx(
                pos[i], self.bounds.min[i], self.bounds.max[i], self._step_sizes[i]
            )
            # TODO : Handle extrapolation
            if temp_idx == -1:
                idx[i] = 0
                toi[i] = 0.0
            else:
                idx[i] = temp_idx
                toi[i] = pos[i] - self.bounds.min[i] - idx[i] * self._step_sizes[i]
        # Interpolation along x-axis
        s0 = lerp(self.values[idx], self.values[idx + tm.ivec2([1, 0])], toi[0])
        s1 = lerp(
            self.values[idx + tm.ivec2([0, 1])],
            self.values[idx + tm.ivec2([1, 1])],
            toi[0],
        )
        # Interpolation along y-axis
        return lerp(s0, s1, toi[1])

    @ti.func
    def _at_3d(self, pos: tm.vec3) -> ti.template():
        idx = tm.ivec3([0, 0, 0])
        toi = tm.vec3([0, 0, 0])
        for i in ti.static(range(3)):
            idx[i] = map_value_to_idx(
                pos[i], self.bounds.min[i], self.bounds.max[i], self._step_sizes[i]
            )
            toi[i] = pos[i] - self.bounds.min[i] - idx[i] * self._step_sizes[i]
        # Interpolation along x-axis
        s0 = lerp(
            self.values[idx],
            self.values[idx + tm.ivec3([1, 0, 0])],
            toi[0],
        )
        s1 = lerp(
            self.values[idx + tm.ivec3([0, 1, 0])],
            self.values[idx + tm.ivec3([1, 1, 0])],
            toi[0],
        )
        s2 = lerp(
            self.values[idx + tm.ivec3([0, 0, 1])],
            self.values[idx + tm.ivec3([1, 0, 1])],
            toi[0],
        )
        s3 = lerp(
            self.values[idx + tm.ivec3([0, 1, 1])],
            self.values[idx + tm.ivec3([1, 1, 1])],
            toi[0],
        )
        # Interpolation along y-axis
        s4 = lerp(s0, s1, toi[1])
        s5 = lerp(s2, s3, toi[1])
        # Interpolation along z-axis
        return lerp(s4, s5, toi[2])

    @ti.func
    def contains(self, pos: ti.template()) -> bool:
        return self.bounds.contains(pos)

    def __getitem__(self, idx: ti.template()) -> float:
        return self.values[idx]

    @property
    def resolution(self):
        return self.values.shape

    @property
    def step_size(self):
        return self._step_sizes


# TODO : Take into account extrapolation
@ti.data_oriented
class ScalarField(SampledField):
    def __init__(
        self,
        values: np.ndarray,
        bounds: Union[Box2D, Box3D],
        extrapolation: Extrapolation = Extrapolation.ZERO,
        needs_grad: bool = False,
    ) -> None:
        ## Dimension and values
        assert (
            values.ndim == 2 or values.ndim == 3
        ), f"Expected sampled field to be 2D or 3D. Get {values.ndim}D"
        self.values = ti.field(dtype=ti.f32, shape=values.shape, needs_grad=needs_grad)
        self.values.from_numpy(values)
        self.dim = values.ndim
        ## Bounds and step sizes
        # assert (isinstance(bounds, Box2D) and self.dim == 2) or (
        #     isinstance(bounds, Box3D) and self.dim == 3
        # ), f"Expected bounds to be an instance of Box{self.dim}D. Get {type(bounds)}"
        self.bounds = bounds

        self._step_sizes = [
            (self.bounds.max[i] - self.bounds.min[i]) / (self.values.shape[i] - 1)
            for i in range(self.dim)
        ]

        assert isinstance(extrapolation, Extrapolation)
        self._extrapolation = extrapolation

    def __repr__(self) -> str:
        return f"ScalarField(values={self.values.shape}, bounds={self.bounds}, extrapolation={self._extrapolation})"


@ti.data_oriented
class VectorField(SampledField):
    def __init__(
        self,
        values: np.ndarray,
        bounds: Union[Box2D, Box3D],
        extrapolation: Extrapolation = Extrapolation.ZERO,
        needs_grad: bool = False,
    ) -> None:
        ## Dimension and values
        dim = values.shape[0]
        assert (dim == 2 and values.ndim == 3) or (
            dim == 3 and values.ndim == 3
        ), f"Expected sampled field to be 2D or 3D vector fields. Get {dim}D"
        self.dim = dim
        self.values = ti.Vector.field(
            dim, dtype=ti.f32, shape=values.shape[1:], needs_grad=needs_grad
        )
        self.values.from_numpy(np.moveaxis(values, 0, -1))

        ## Bounds and step sizes
        # assert (isinstance(bounds, Box2D) and self.dim == 2) or (
        #     isinstance(bounds, Box3D) and self.dim == 3
        # ), f"Expected bounds to be an instance of Box{self.dim}D. Get {type(bounds)}"
        self.bounds = bounds
        self._step_sizes = [
            (self.bounds.max[i] - self.bounds.min[i]) / (self.values.shape[i] - 1)
            for i in range(self.dim)
        ]

        assert isinstance(extrapolation, Extrapolation)
        self._extrapolation = extrapolation

    # def at(self, pos: ti.template()) -> ti.template():
    #     if self.dim == 2:
    #         value = self._at_2d(pos)
    #     elif self.dim == 3:
    #         value = self._at_3d(pos)
    #     else:
    #         raise NotImplementedError
    #     return value

    # @ti.func
    # def _at_2d(self, pos: tm.vec2) -> tm.vec2:
    #     idx = tm.ivec2([0, 0])
    #     toi = tm.vec2([0, 0])
    #     for i in ti.static(range(2)):
    #         idx[i] = map_value_to_idx(
    #             pos[i], self._bounds.min[i], self._bounds.max[i], self._step_sizes[i]
    #         )
    #         toi[i] = pos[i] - self._bounds.min[i] - idx[i] * self._step_sizes[i]
    #     # Interpolation along x-axis
    #     s0 = lerp(self.values[idx], self.values[idx + tm.ivec2([1, 0])], toi[0])
    #     s1 = lerp(
    #         self.values[idx + tm.ivec2([0, 1])],
    #         self.values[idx + tm.ivec2([1, 1])],
    #         toi[0],
    #     )
    #     # Interpolation along y-axis
    #     return lerp(s0, s1, toi[1])

    # @ti.func
    # def _at_3d(self, pos: tm.vec3) -> tm.vec3:
    #     idx = tm.ivec3([0, 0, 0])
    #     toi = tm.vec3([0, 0, 0])
    #     for i in ti.static(range(3)):
    #         idx[i] = map_value_to_idx(
    #             pos[i], self._bounds.min[i], self._bounds.max[i], self._step_sizes[i]
    #         )
    #         toi[i] = pos[i] - self._bounds.min[i] - idx[i] * self._step_sizes[i]
    #     # Interpolation along x-axis
    #     s0 = lerp(
    #         self.values[idx],
    #         self.values[idx + tm.ivec3([1, 0, 0])],
    #         toi[0],
    #     )
    #     s1 = lerp(
    #         self.values[idx + tm.ivec3([0, 1, 0])],
    #         self.values[idx + tm.ivec3([1, 1, 0])],
    #         toi[0],
    #     )
    #     s2 = lerp(
    #         self.values[idx + tm.ivec3([0, 0, 1])],
    #         self.values[idx + tm.ivec3([1, 0, 1])],
    #         toi[0],
    #     )
    #     s3 = lerp(
    #         self.values[idx + tm.ivec3([0, 1, 1])],
    #         self.values[idx + tm.ivec3([1, 1, 1])],
    #         toi[0],
    #     )
    #     # Interpolation along y-axis
    #     s4 = lerp(s0, s1, toi[1])
    #     s5 = lerp(s2, s3, toi[1])
    #     # Interpolation along z-axis
    #     return lerp(s4, s5, toi[2])

    @property
    @ti.kernel
    def max(self) -> float:
        max_val = -tm.inf
        for I in ti.grouped(self.values):
            max_val = ti.max(max_val, tm.length(self.values[I]))
        return max_val

    def __repr__(self) -> str:
        return f"VectorField(values={self.values.shape}, bounds={self.bounds}, extrapolation={self._extrapolation})"


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
    values = scalar_field.values.to_numpy()

    grad_values = np.stack(
        np.gradient(values, *scalar_field.step_size, edge_order=2),
        dtype=np.float32,
    )
    return VectorField(
        grad_values,
        scalar_field.bounds,
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
    values = scalar_field.values.to_numpy()
    log_values = np.log(np.where(values > eps, values, eps))
    return ScalarField(
        log_values,
        scalar_field.bounds,
        scalar_field._extrapolation,
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
    print(vector_field.max, np.max(np.linalg.norm(vector_field.values, axis=-1)))

    plt.imshow(values, origin="lower")
    plt.quiver(vector_field.values[:, :, 0], vector_field.values[:, :, 1], color="k")
    plt.show()
