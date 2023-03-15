from snake_ai.envs.snake import Snake, SnakeAI, SnakeHuman
from snake_ai.envs.geometry import Rectangle
from snake_ai.utils import Direction, errors
import pytest

from snake_ai.envs.walker import Walker2D
class TestWalker2D:
    x, y, pixel = 2, 1, 10
    walker = Walker2D(x, y, pixel)
        
    def test_init(self):
        assert self.walker.position == Rectangle(self.x * self.pixel, self.y * self.pixel, self.pixel, self.pixel)
        assert self.walker.grid_position == (2,1)
        assert self.walker.pixel == 10
        assert self.walker.direction == Direction.NORTH

        with pytest.raises(ValueError):
            walker = Walker2D(self.x, self.y, -5)
        
    def test_position_setter(self):
        new_pos = Rectangle(10, 20, self.pixel, self.pixel)
        self.walker.position = new_pos
        assert self.walker.position == new_pos
        assert self.walker.grid_position == (1, 2)
        
        with pytest.raises(ValueError):
            self.walker.position = Rectangle(10, 20, 2 * self.pixel, 0.5 * self.pixel)

    def test_move(self):
        self.walker.move(Direction.NORTH)
        assert self.walker._direction == Direction.NORTH
        assert self.walker.grid_position == (self.x, self.y - 1)
        self.walker.move(Direction.EAST)
        assert self.walker._direction == Direction.EAST
        assert self.walker.grid_position == (self.x + 1, self.y - 1)
        self.walker.move(Direction.SOUTH)
        assert self.walker._direction == Direction.SOUTH
        assert self.walker.grid_position == (self.x + 1, self.y)
        self.walker.move(Direction.WEST)
        assert self.walker._direction == Direction.WEST
        assert self.walker.grid_position == (self.x, self.y)
        
        with pytest.raises(errors.UnknownDirection):
            self.walker.move(Direction.NORTH_WEST)

    def test_move_from_action(self):
        self.walker.move_from_action(0)
        assert self.walker._direction == Direction.NORTH
        assert self.walker.grid_position == (self.x, self.y - 1)
        self.walker.move_from_action(1)
        assert self.walker._direction == Direction.EAST
        assert self.walker.grid_position == (self.x + 1, self.y - 1)
        self.walker.move_from_action(2)
        assert self.walker._direction == Direction.SOUTH
        assert self.walker.grid_position == (self.x + 1, self.y)
        self.walker.move_from_action(3)
        assert self.walker._direction == Direction.WEST
        assert self.walker.grid_position == (self.x, self.y)
        
        with pytest.raises(ValueError):
            self.walker.move_from_action(-2)
            
    def test_neighbours(self):
        top, right, bottom, left = self.walker.neighbours
        assert top == Rectangle(self.x * self.pixel, (self.y - 1) * self.pixel, self.pixel, self.pixel)
        assert right == Rectangle((self.x + 1) * self.pixel, self.y * self.pixel, self.pixel, self.pixel)
        assert bottom == Rectangle(self.x * self.pixel, (self.y + 1) * self.pixel, self.pixel, self.pixel)
        assert left == Rectangle((self.x - 1) * self.pixel, self.y * self.pixel, self.pixel, self.pixel)

    def test_eq(self):
        same_walker = Walker2D(self.x, self.y, self.pixel)
        other_walker = Walker2D(self.x + 10, self.y, self.pixel)
        other_walker_2 = Walker2D(self.x, self.y, self.pixel + 5)
        other_walker_3 = Walker2D(self.x, self.y, self.pixel)
        other_walker_3.direction = Direction.SOUTH
        assert same_walker == self.walker
        assert other_walker != self.walker
        assert other_walker_2 != self.walker
        assert other_walker_3 != self.walker

        with pytest.raises(TypeError):
            same_walker == Snake([(10,10), (10, 11)], self.pixel)

from snake_ai.envs.snake import Snake, check_snake_positions
class TestSnake:
    pixel = 10
    snake = Snake([(0,0), (1, 0), (2, 0)], pixel)

    def test_position_checker(self):
        valid_positions = [(1,0), (1,1), (2,1), (2,2)]
        check_snake_positions(valid_positions)
        
        invalid_type = tuple(valid_positions)
        with pytest.raises(TypeError):
            check_snake_positions(invalid_type)

        invalid_type2 = [(0,0), [1,2], (1,2,3)]
        with pytest.raises(TypeError):
            check_snake_positions(invalid_type2)

        invalid_length = [(0,0)]
        with pytest.raises(errors.ShapeError):
            check_snake_positions(invalid_length)
        # The third position has a displacement vector equal to (2, 0)
        degenerated = [(0,0), (0,1), (2,1), (2,2)] 
        with pytest.raises(errors.ConfigurationError):
            check_snake_positions(degenerated)
        # The third position is the same as the second
        degenerated = [(0,0), (0,1), (0,1), (2,2)] 
        with pytest.raises(errors.ConfigurationError):
            check_snake_positions(degenerated)
    
    def test_init(self):
        assert self.snake.head == self.snake.position
        assert self.snake.head == Rectangle(0, 0, self.pixel, self.pixel)
        assert self.snake.body == [Rectangle(self.pixel, 0, self.pixel, self.pixel), 
                                   Rectangle(2 * self.pixel, 0, self.pixel, self.pixel),]
        assert self.snake.direction == Direction.WEST
        assert self.snake.pixel == self.pixel
        assert len(self.snake) == 3
        # Check each possible orientation of the snake 
        east_snake = Snake([(0,0), (-1,0)], self.pixel)
        assert east_snake.direction == Direction.EAST
        north_snake = Snake([(0,0), (0,1)], self.pixel)
        assert north_snake.direction == Direction.NORTH
        west_snake = Snake([(0,0), (1,0)], self.pixel)
        assert west_snake.direction == Direction.WEST
        south_snake = Snake([(0,0), (0,-1)], self.pixel)
        assert south_snake.direction == Direction.SOUTH

        with pytest.raises(errors.ConfigurationError):
            Snake([(0,0), (1,0),(1,1),(0,1), (0,0)], self.pixel)
        
    def test_move(self):
        self.snake.move(Direction.WEST)
        snake_west = Snake([(-1, 0), (0, 0), (1, 0)], self.pixel)
        assert self.snake == snake_west
        self.snake.move(Direction.NORTH)
        snake_north = Snake([(-1, -1), (-1, 0), (0, 0)], self.pixel)
        assert self.snake == snake_north
        self.snake.move(Direction.EAST)
        snake_east = Snake([(0, -1), (-1, -1), (-1, 0)], self.pixel)
        assert self.snake == snake_east
        self.snake.move(Direction.SOUTH)
        snake_south = Snake([(0, 0), (0, -1), (-1, -1)], self.pixel)
        assert self.snake == snake_south
        
        with pytest.raises(errors.CollisionError):
            self.snake.move(Direction.NORTH)

    def test_move_from_action(self):
        # left turn
        self.snake.move_from_action(0)
        snake_left = Snake([(0, 1), (0, 0), (1, 0)], self.pixel)
        assert self.snake == snake_left
        # continue straight
        self.snake.move_from_action(1)
        snake_straight = Snake([(0, 2), (0, 1), (0, 0)], self.pixel)
        assert self.snake == snake_straight
        # right turn
        self.snake.move_from_action(2)
        snake_right = Snake([(-1, 2), (0, 2), (0, 1)], self.pixel)
        assert self.snake == snake_right

        with pytest.raises(AssertionError):
            self.snake.move_from_action(3)

    def test_grow(self):
        east_snake = Snake([(0,0), (-1,0)], self.pixel)
        east_snake.grow()
        assert east_snake == Snake([(0,0), (-1,0), (-2,0)], self.pixel)
        north_snake = Snake([(0,0), (0,1)], self.pixel)
        north_snake.grow()
        assert north_snake == Snake([(0,0), (0,1), (0,2)], self.pixel)
        west_snake = Snake([(0,0), (1,0)], self.pixel)
        west_snake.grow()
        assert west_snake == Snake([(0,0), (1,0), (2,0)], self.pixel)
        south_snake = Snake([(0,0), (0,-1)], self.pixel)
        south_snake.grow()
        assert south_snake == Snake([(0,0), (0,-1), (0,-2)], self.pixel)

    def test_collision(self):
        assert not self.snake.collide_with_itself()
        colliding_snake = Snake([(0,0), (1,0),(1,1),(0,1)], self.pixel)
        colliding_snake.grow()
        colliding_snake.move(Direction.SOUTH)
        assert colliding_snake.collide_with_itself()

    def test_for_loop(self):
        for i, part in enumerate(self.snake):
            if i == 0:
                assert part == self.snake.head
                assert part == self.snake.position
            else:
                assert part == self.snake.body[i-1]

    def test_neighbours(self):
        north = Rectangle(0, -self.pixel, self.pixel, self.pixel)
        east = Rectangle(self.pixel, 0, self.pixel, self.pixel)
        south = Rectangle(0, self.pixel, self.pixel, self.pixel)
        west = Rectangle(-self.pixel, 0, self.pixel, self.pixel)
        # West
        left, front, right = self.snake.neighbours
        assert left == south
        assert front == west
        assert right == north
        # North
        self.snake.direction = Direction.NORTH
        left, front, right = self.snake.neighbours
        assert left == west
        assert front == north
        assert right == east
        # East
        self.snake.direction = Direction.EAST
        left, front, right = self.snake.neighbours
        assert left == north
        assert front == east
        assert right == south
        # South
        self.snake.direction = Direction.SOUTH
        left, front, right = self.snake.neighbours
        assert left == east
        assert front == south
        assert right == west
        
# class TestSnake:
#     x, y = 0, 0
#     pix = 10
#     snake = Snake(x, y, pix)

#     def test_for(self):
#         for i, part in enumerate(self.snake):
#             offset = i * self.pix
#             assert part == pygame.Rect(self.x - offset, self.y, self.pix, self.pix)

#     def test_equal(self):
#         assert self.snake == Snake(self.x, self.y, self.pix)

#     def test_dictionary_conversion(self):
#         dictionary = self.snake.to_dict()
#         assert dictionary == {'x' : self.x, 'y' : self.y, 'pixel' : self.pix}

#         new_snake = Snake.from_dict(dictionary)
#         assert new_snake == self.snake