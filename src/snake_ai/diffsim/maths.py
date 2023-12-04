import taichi as ti


@ti.func
def lerp(a: float, b: float, t: float):
    return a + (b - a) * t
