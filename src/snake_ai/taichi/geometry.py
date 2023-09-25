import numpy as np
import taichi as ti
import taichi.math as tm

from snake_ai.envs import Rectangle
from typing import List


def convert_rectangle(rect: Rectangle):
    return Box2D(
        tm.vec2(rect.x, rect.y), tm.vec2(rect.x + rect.width, rect.y + rect.height)
    )


def convert_rectangles(rectangles: List[Rectangle]) -> ti.Field:
    """Convert a list of rectangles to a Box2D field.

    Args:
        rectangles (List[Rectangle]): List of rectangles.

    Returns:
        ti.Field: A Box2D field with the same number of rectangles.
    """
    if len(rectangles) == 0:
        raise ValueError("Expected at least one rectangle.")
    assert all(
        isinstance(rect, Rectangle) for rect in rectangles
    ), "Expected obstacles to be a list of Rectangle. Get {}".format(
        [type(rect) for rect in rectangles]
    )
    boxes = Box2D.field(shape=len(rectangles))
    min_values = np.array([[rect.x, rect.y] for rect in rectangles])
    max_values = np.array(
        [[rect.x + rect.width, rect.y + rect.height] for rect in rectangles]
    )
    boxes.min.from_numpy(min_values)
    boxes.max.from_numpy(max_values)
    return boxes


@ti.dataclass
class Box2D:
    min: tm.vec2
    max: tm.vec2

    def width(self):
        return self.max[0] - self.min[0]

    def height(self):
        return self.max[1] - self.min[1]

    @ti.func
    def contains(self, pos: ti.math.vec2) -> bool:
        clamped_pos = tm.clamp(pos, self.min, self.max)
        return clamped_pos.x == pos.x and clamped_pos.y == pos.y


@ti.dataclass
class Box3D:
    min: tm.vec3
    max: tm.vec3

    def width(self):
        return self.max[0] - self.min[0]

    def height(self):
        return self.max[1] - self.min[1]

    def depth(self):
        return self.max[2] - self.min[2]

    @ti.func
    def contains(self, pos: ti.math.vec3) -> bool:
        clamped_pos = tm.clamp(pos, self.min, self.max)
        return (
            clamped_pos.x == pos.x and clamped_pos.y == pos.y and clamped_pos.z == pos.z
        )


if __name__ == "__main__":
    rect = Rectangle(0, 0, 1, 1)
    rect2 = Rectangle(1, 1, 2, 2)
    box = convert_rectangle(rect)
    assert box.min == tm.vec2(0, 0)
    assert box.max == tm.vec2(1, 1)
    ti.init(arch=ti.cpu)

    boxes = convert_rectangles([rect, rect2])
    boxes_dict = boxes.to_numpy()
    assert np.equal(
        boxes_dict["min"], np.array([[0, 0], [1, 1]])
    ).all(), f"The minimum values are {boxes_dict['min']}"
    assert np.equal(
        boxes_dict["max"], np.array([[1, 1], [3, 3]])
    ).all(), f"The maximum values are {boxes_dict['max']}"

    @ti.kernel
    def test():
        print(boxes[0].contains(tm.vec2(0.5, 0.5)))

    print("Hello world!")
    test()
