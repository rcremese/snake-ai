from snake_ai.envs import GridWorld, MazeGrid
from phi import flow
from snake_ai.envs import Rectangle
import pytest

from snake_ai.physim.converter import DiffusionConverter
class TestDiffusionConverter:
    w, h, pixel = 20, 20, 10
    goal = Rectangle(0, 0, pixel, pixel)

    env = GridWorld(w, h, pixel)
    env.reset()
    env.goal = goal

    def test_init(self):
        converter = DiffusionConverter("meta", 1)
        assert converter.type == "meta"
        assert converter._init_value == 1

        converter = DiffusionConverter("pixel", 1e6)
        assert converter.type == "pixel"
        assert converter._init_value == 1e6

        with pytest.raises(ValueError):
            DiffusionConverter("meta", -1)
        with pytest.raises(ValueError):
            DiffusionConverter("space", 10)

    def test_convertion(self):
        meta_converter = DiffusionConverter("meta", 1)
        init_distrib, obstacles = meta_converter(self.env)
        assert init_distrib == flow.CenteredGrid(flow.Box(x=(0, 1), y=(0, 1)), bounds=flow.Box(x=self.w, y=self.h), x=self.w, y=self.h)
        assert obstacles == []

        pixel_converter = DiffusionConverter("pixel", 1e6)
        init_distrib, obstacles = pixel_converter(self.env)
        assert init_distrib == 1e6 * flow.CenteredGrid(flow.Sphere(x=self.pixel // 2, y=self.pixel // 2, radius=self.pixel // 2),
                                                 bounds=flow.Box(x=self.w * self.pixel, y=self.h * self.pixel),
                                                 x=self.w * self.pixel, y=self.h * self.pixel)
        assert obstacles == []

