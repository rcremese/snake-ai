from snake_ai.envs.snake import Snake, SnakeAI, SnakeHuman
import pygame

class TestSnake:
    x, y = 0, 0
    pix = 10
    snake = Snake(x, y, pix)

    def test_for(self):
        for i, part in enumerate(self.snake):
            offset = i * self.pix
            assert part == pygame.Rect(self.x - offset, self.y, self.pix, self.pix)

    def test_equal(self):
        assert self.snake == Snake(self.x, self.y, self.pix)

    def test_dictionary_conversion(self):
        dictionary = self.snake.to_dict()
        assert dictionary == {'x' : self.x, 'y' : self.y, 'pixel' : self.pix}

        new_snake = Snake.from_dict(dictionary)
        assert new_snake == self.snake