from snake_ai.envs import GridWorld, RandomObstaclesEnv, MazeGrid
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
        converter = DiffusionConverter("meta")
        assert converter.type == "meta"
        converter = DiffusionConverter("pixel")
        assert converter.type == "pixel"

        with pytest.raises(ValueError):
            DiffusionConverter("space")

    def test_convertion(self):
        meta_converter = DiffusionConverter("meta")
        init_distrib = meta_converter(self.env)
        assert init_distrib == flow.CenteredGrid(flow.Box(x=(0, 1), y=(0, 1)), bounds=flow.Box(x=self.w, y=self.h), x=self.w, y=self.h)

        pixel_converter = DiffusionConverter("pixel")
        init_distrib = pixel_converter(self.env)
        assert init_distrib == flow.CenteredGrid(flow.Sphere(x=self.pixel // 2, y=self.pixel // 2, radius=self.pixel // 2),
                                                 bounds=flow.Box(x=self.w * self.pixel, y=self.h * self.pixel),
                                                 x=self.w * self.pixel, y=self.h * self.pixel)

from snake_ai.physim.converter import ObstacleConverter
class TestObstacleConverter:
    w, h, pixel = 20, 20, 10
    obstacles = [Rectangle(0, 0, pixel, pixel), Rectangle(10 * pixel, 10 * pixel, 2 * pixel, 2 * pixel)]

    env = RandomObstaclesEnv(w, h, pixel)
    env.reset()
    env.obstacles = obstacles

    def test_convertion(self):
        meta_converter = ObstacleConverter("meta")
        obstacle_grid = meta_converter(self.env)
        obstacle_metapixel = [flow.Box(x=(0, 1), y=(0, 1)), flow.Box(x=(10, 12), y=(10, 12))]
        assert obstacle_grid == flow.CenteredGrid(flow.union(obstacle_metapixel), x=self.w, y=self.h)

        pixel_converter = ObstacleConverter("pixel")
        obstacle_grid = pixel_converter(self.env)
        obstacle_pixel = [flow.Box(x=(0, self.pixel), y=(0, self.pixel)), flow.Box(x=(10 * self.pixel, 12 * self.pixel), y=(10 * self.pixel, 12 * self.pixel))]
        assert obstacle_grid == flow.CenteredGrid(flow.union(obstacle_pixel), x=self.w * self.pixel, y=self.h * self.pixel)

        # Without obstacles
        self.env.obstacles = []
        obstacle_grid = meta_converter(self.env)
        assert obstacle_grid == flow.CenteredGrid(flow.math.zeros(flow.math.spatial(x=self.w, y = self.h)))
        obstacle_grid = pixel_converter(self.env)
        assert obstacle_grid == flow.CenteredGrid(flow.math.zeros(flow.math.spatial(x=self.w * self.pixel, y = self.h * self.pixel)))

from snake_ai.physim.converter import PointCloudConverter
class TestPointCloudConverter:
    w, h, pixel = 5, 5, 10
    goal = Rectangle(4 * pixel, 0, pixel, pixel)

    env = MazeGrid(w, h, pixel, seed=0)
    env.reset()
    # Maze obstained with the seed 0
    #  [A 1 0 1 G]
    #  [0 1 0 1 0]
    #  [0 0 0 0 0]
    #  [0 1 0 1 0]
    #  [0 1 0 1 0]
    env.goal = goal

    def test_convertion(self):
        meta_converter = PointCloudConverter("meta")
        point_cloud = meta_converter(self.env)
        points = [flow.vec(x=0, y=0), flow.vec(x=0, y=1), flow.vec(x=0, y=2), flow.vec(x=0, y=3), flow.vec(x=0, y=4),
                  flow.vec(x=1, y=2),
                  flow.vec(x=2, y=0), flow.vec(x=2, y=1), flow.vec(x=2, y=2), flow.vec(x=2, y=3), flow.vec(x=2, y=4),
                  flow.vec(x=3, y=2),
                  flow.vec(x=4, y=1), flow.vec(x=4, y=2), flow.vec(x=4, y=3), flow.vec(x=4, y=4)]
        assert all(point_cloud == flow.tensor(points, flow.instance('point')))

        pixel_converter = PointCloudConverter("pixel")
        point_cloud = pixel_converter(self.env)
        pixel_points = [(point + 0.5) * self.pixel for point in points]
        assert all(point_cloud == flow.tensor(pixel_points, flow.instance('point')))