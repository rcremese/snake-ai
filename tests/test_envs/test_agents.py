from snake_ai.envs.geometry import Rectangle
from snake_ai.utils import Direction, errors
import pytest

from snake_ai.envs.walker import Walker2D
class TestWalker2D:
    x, y, pixel = 2, 1, 10

    def test_init(self):
        walker = Walker2D(self.x, self.y, self.pixel)
        assert walker.position == Rectangle(self.x * self.pixel, self.y * self.pixel, self.pixel, self.pixel)
        assert walker.pixel == 10
        assert walker.direction == Direction.NORTH

        with pytest.raises(ValueError):
            Walker2D(self.x, self.y, -5)

    def test_position_setter(self):
        walker = Walker2D(self.x, self.y, self.pixel)

        new_pos = Rectangle(10, 20, self.pixel, self.pixel)
        walker.position = new_pos
        assert walker.position == new_pos

        with pytest.raises(ValueError):
            walker.position = Rectangle(10, 20, 2 * self.pixel, 0.5 * self.pixel)

    def test_move(self):
        walker = Walker2D(self.x, self.y, self.pixel)

        walker.move(Direction.NORTH)
        assert walker._direction == Direction.NORTH
        assert walker.position == Rectangle(self.x * self.pixel, (self.y - 1) * self.pixel, self.pixel, self.pixel)
        walker.move(Direction.EAST)
        assert walker._direction == Direction.EAST
        assert walker.position == Rectangle((self.x + 1) * self.pixel, (self.y - 1) * self.pixel, self.pixel, self.pixel)
        walker.move(Direction.SOUTH)
        assert walker._direction == Direction.SOUTH
        assert walker.position == Rectangle((self.x + 1) * self.pixel, self.y * self.pixel, self.pixel, self.pixel)
        walker.move(Direction.WEST)
        assert walker._direction == Direction.WEST
        assert walker.position == Rectangle(self.x * self.pixel, self.y * self.pixel, self.pixel, self.pixel)

        with pytest.raises(errors.UnknownDirection):
            walker.move(Direction.NORTH_WEST)

    def test_move_from_action(self):
        walker = Walker2D(self.x, self.y, self.pixel)

        walker.move_from_action(0)
        assert walker._direction == Direction.NORTH
        assert walker.position == Rectangle(self.x * self.pixel, (self.y - 1) * self.pixel, self.pixel, self.pixel)
        walker.move_from_action(1)
        assert walker._direction == Direction.EAST
        assert walker.position == Rectangle((self.x + 1) * self.pixel, (self.y - 1) * self.pixel, self.pixel, self.pixel)
        walker.move_from_action(2)
        assert walker._direction == Direction.SOUTH
        assert walker.position == Rectangle((self.x + 1) * self.pixel, self.y * self.pixel, self.pixel, self.pixel)
        walker.move_from_action(3)
        assert walker._direction == Direction.WEST
        assert walker.position == Rectangle(self.x * self.pixel, self.y * self.pixel, self.pixel, self.pixel)

        with pytest.raises(ValueError):
            walker.move_from_action(-2)

    def test_neighbours(self):
        walker = Walker2D(self.x, self.y, self.pixel)

        top, right, bottom, left = walker.neighbours
        assert top == Rectangle(self.x * self.pixel, (self.y - 1) * self.pixel, self.pixel, self.pixel)
        assert right == Rectangle((self.x + 1) * self.pixel, self.y * self.pixel, self.pixel, self.pixel)
        assert bottom == Rectangle(self.x * self.pixel, (self.y + 1) * self.pixel, self.pixel, self.pixel)
        assert left == Rectangle((self.x - 1) * self.pixel, self.y * self.pixel, self.pixel, self.pixel)

    def test_eq(self):
        walker = Walker2D(self.x, self.y, self.pixel)
        same_walker = Walker2D(self.x, self.y, self.pixel)
        other_walker = Walker2D(self.x + 10, self.y, self.pixel)
        other_walker_2 = Walker2D(self.x, self.y, self.pixel + 5)
        other_walker_3 = Walker2D(self.x, self.y, self.pixel)
        other_walker_3.direction = Direction.SOUTH
        assert same_walker == walker
        assert other_walker != walker
        assert other_walker_2 != walker
        assert other_walker_3 != walker

        with pytest.raises(TypeError):
            same_walker == Snake([(10,10), (10, 11)], self.pixel)

from snake_ai.envs.snake import Snake, check_snake_positions, get_direction_from_positions
class TestSnake:
    pixel = 10
    positions = [(0,0), (1, 0), (2, 0)]

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

    def test_direction_from_position(self):
        snake = Snake(self.positions, self.pixel)

        assert get_direction_from_positions(snake.head, snake.body[0]) == Direction.WEST
        assert get_direction_from_positions(snake.body[-1], snake.body[-2]) == Direction.EAST
        snake.move(Direction.NORTH)
        assert get_direction_from_positions(snake.head, snake.body[0]) == Direction.NORTH
        snake.move(Direction.NORTH)
        assert get_direction_from_positions(snake.body[-1], snake.body[-2]) == Direction.SOUTH


    def test_init(self):
        snake = Snake(self.positions, self.pixel)

        assert snake.head == snake.position
        assert snake.head == Rectangle(0, 0, self.pixel, self.pixel)
        assert snake.body == [Rectangle(self.pixel, 0, self.pixel, self.pixel),
                                   Rectangle(2 * self.pixel, 0, self.pixel, self.pixel),]
        assert snake.direction == Direction.WEST
        assert snake.pixel == self.pixel
        assert len(snake) == 3
        # Check each possible orientation of snake
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
        snake = Snake(self.positions, self.pixel)

        snake.move(Direction.WEST)
        snake_west = Snake([(-1, 0), (0, 0), (1, 0)], self.pixel)
        assert snake == snake_west
        snake.move(Direction.NORTH)
        snake_north = Snake([(-1, -1), (-1, 0), (0, 0)], self.pixel)
        assert snake == snake_north
        snake.move(Direction.EAST)
        snake_east = Snake([(0, -1), (-1, -1), (-1, 0)], self.pixel)
        assert snake == snake_east
        snake.move(Direction.SOUTH)
        snake_south = Snake([(0, 0), (0, -1), (-1, -1)], self.pixel)
        assert snake == snake_south

        with pytest.raises(errors.CollisionError):
            snake.move(Direction.NORTH)

    def test_position_setter(self):
        snake = Snake(self.positions, self.pixel)
        new_head = Rectangle(2 * self.pixel, self.pixel, self.pixel, self.pixel)
        snake.position = new_head
        assert snake.position == new_head == snake.head
        assert snake == Snake([(2,1), (3,1), (4,1)], self.pixel)

    def test_move_from_action(self):
        snake = Snake(self.positions, self.pixel)
        # left turn
        snake.move_from_action(0)
        snake_left = Snake([(0, 1), (0, 0), (1, 0)], self.pixel)
        assert snake == snake_left
        # continue straight
        snake.move_from_action(1)
        snake_straight = Snake([(0, 2), (0, 1), (0, 0)], self.pixel)
        assert snake == snake_straight
        # right turn
        snake.move_from_action(2)
        snake_right = Snake([(-1, 2), (0, 2), (0, 1)], self.pixel)
        assert snake == snake_right

        with pytest.raises(errors.InvalidAction):
            snake.move_from_action(3)

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
        snake = Snake(self.positions, self.pixel)
        assert not snake.collide_with_itself()
        colliding_snake = Snake([(0,0), (1,0),(1,1),(0,1)], self.pixel)
        colliding_snake.grow()
        colliding_snake.move(Direction.SOUTH)
        assert colliding_snake.collide_with_itself()

    def test_for_loop(self):
        snake = Snake(self.positions, self.pixel)
        for i, part in enumerate(snake):
            if i == 0:
                assert part == snake.head
                assert part == snake.position
            else:
                assert part == snake.body[i-1]

    def test_neighbours(self):
        snake = Snake(self.positions, self.pixel)
        north = Rectangle(0, -self.pixel, self.pixel, self.pixel)
        east = Rectangle(self.pixel, 0, self.pixel, self.pixel)
        south = Rectangle(0, self.pixel, self.pixel, self.pixel)
        west = Rectangle(-self.pixel, 0, self.pixel, self.pixel)
        # West
        left, front, right = snake.neighbours
        assert left == south
        assert front == west
        assert right == north
        # North
        snake.direction = Direction.NORTH
        left, front, right = snake.neighbours
        assert left == west
        assert front == north
        assert right == east
        # East
        snake.direction = Direction.EAST
        left, front, right = snake.neighbours
        assert left == north
        assert front == east
        assert right == south
        # South
        snake.direction = Direction.SOUTH
        left, front, right = snake.neighbours
        assert left == east
        assert front == south
        assert right == west

from snake_ai.envs.snake import BidirectionalSnake
class TestBidirectionalSnake:
    sn_positions = [(0,0),(0,1),(0,2)]
    ew_positions = [(0,0),(1,0),(2,0)]
    pixel = 10

    def test_go_back(self):
        snake_sn = BidirectionalSnake(self.sn_positions, self.pixel)
        assert snake_sn.direction == Direction.NORTH
        snake_sn.go_back()
        assert snake_sn.direction == Direction.SOUTH
        assert snake_sn == Snake(self.sn_positions[::-1], self.pixel)

        snake_ew = BidirectionalSnake(self.ew_positions, self.pixel)
        assert snake_ew.direction == Direction.WEST
        snake_ew.go_back()
        assert snake_ew.direction == Direction.EAST
        assert snake_ew == Snake(self.ew_positions[::-1], self.pixel)

    def test_move_from_action(self):
        snake = BidirectionalSnake(self.ew_positions, self.pixel)
        # left turn
        snake.move_from_action(0)
        snake_left = Snake([(0, 1), (0, 0), (1, 0)], self.pixel)
        assert snake == snake_left
        # continue straight
        snake.move_from_action(1)
        snake_straight = Snake([(0, 2), (0, 1), (0, 0)], self.pixel)
        assert snake == snake_straight
        # right turn
        snake.move_from_action(2)
        snake_right = Snake([(-1, 2), (0, 2), (0, 1)], self.pixel)
        assert snake == snake_right
        # turn_back
        snake.move_from_action(3)
        assert snake == Snake([(0, 1), (0, 2), (-1, 2)], self.pixel)

        with pytest.raises(AssertionError):
            snake.move_from_action(4)
