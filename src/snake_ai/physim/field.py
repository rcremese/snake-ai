import taichi as ti
import taichi.math as tm

@ti.dataclass
class SampledField:
    def __init__(self) -> None:
        self.values = ti.Vector.field(2, dtype=ti.f32)