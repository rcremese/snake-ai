from snake_ai.envs.geometry import Rectangle, Circle
from phi import flow
import pygame

class TestRectangle:
    x_0, y_0, w_0, h_0 = 0, 0, 1, 1
    x_1, y_1, w_1, h_1 = -5, -4, -1, 2

    def test_init(self):
        rect_1 = Rectangle(self.x_0, self.y_0, self.w_0, self.h_0)
        rect_2 = Rectangle(pygame.Rect(self.x_0, self.y_0, self.w_0, self.h_0))
        assert rect_1 == rect_2

    def test_phiflow_conversion(self):
        rect = Rectangle(self.x_0, self.y_0, self.w_0, self.h_0)
        phiflow_rect = rect.to_phiflow()
        assert phiflow_rect == flow.Box(x=(self.x_0, self.x_0 + self.w_0), y=(self.y_0, self.y_0 + self.h_0))

    def test_dict_conversion(self):
        rect = Rectangle(self.x_0, self.y_0, self.w_0, self.h_0)
        dictionary = {'left' : self.x_0, 'right' : self.x_0 + self.w_0, 'top' : self.y_0, 'bottom' : self.y_0 + self.h_0}
        rect_dict = rect.to_dict()
        assert rect_dict == dictionary

        new_rect = Rectangle.from_dict(dictionary)
        assert new_rect == rect

class TestSphere:
    x, y = 0, 1
    radius = 1
    sphere = Circle(x, y, radius)

    def test_dict_conversion(self):
        dictionary = self.sphere.to_dict()
        assert dictionary == {'center' : [self.x, self.y], 'radius': self.radius}
        new_sphere = Circle.from_dict(dictionary)
        assert new_sphere == self.sphere