##
# @author  <>
 # @file Description
 # @desc Created on 2023-03-09 5:52:59 pm
 # @copyright
 #
import gym
import pygame
from snake_ai.envs.snake import Snake, SnakeAI
from snake_ai.envs.grid_world import GridWorld
from snake_ai.envs.geometry import Rectangle
from snake_ai.utils import Colors, Reward, Direction
from snake_ai.utils.errors import CollisionError, ConfigurationError, InitialisationError
from typing import Optional, Tuple, Dict, List
import numpy as np

class SnakeEnv(gym.Env, GridWorld):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, width: int = 20, height: int = 20, nb_obstacles: int = 0, pixel: int = 20,
                 max_obs_size: int = 1, seed: int = 0, render_mode : Optional[str]=None):
        super().__init__(width, height, pixel, nb_obstacles, max_obs_size)
        # Construct a list with the number of obstacle samples per size (first value = 1 metapixel).
        self._nb_samples_per_size = self._max_obs_size * [self.nb_obs // self._max_obs_size, ]
        # If the number of obstacles can not be devided by the maximum value, samples are added to the obstacles of size 1
        self._nb_samples_per_size[0] += self.nb_obs - sum(self._nb_samples_per_size)

        if (render_mode is not None) and (render_mode not in self.metadata["render_modes"]):
            raise ValueError(f"render_mode should be eather None or in {self.metadata['render_modes']}. Get {render_mode}")
        self.render_mode = render_mode

        if not isinstance(seed, int):
            raise TypeError(f"Seed must be an integer, not {type(seed)}")
        self._seed = seed

        # All non-instanciated attributes
        self._snake = None
        self._screen = None
        self._free_position_mask = None

        if self.render_mode == "human":
            self._init_human_renderer()

    def reset(self, seed : Optional[int]=None):
        self.seed(seed)
        # Initialise a grid of free positions and score
        self.score = 0
        # self.truncated = False
        self._free_position_mask = np.ones((self.width, self.height))
        # Initialise obstacles
        self._obstacles = []
        if self.nb_obs > 0:
            self._obstacles = self._populate_grid_with_obstacles()
        # Initialise food & snake
        self._goal = self._place_food()
        self._snake = self._place_snake()
        self._position = self._snake.head

    def step(self, action : int) -> Tuple[np.ndarray, Reward, bool, Dict]:
        self._snake.move_from_action(action)
        self._position = self._snake.head

        truncated = self._position.colliderect(self._goal)
        # A flag is set if the snake has reached the food
        # An other one if the snake is outside or collide with obstacles
        terminated = self._is_collision()
        # Give a reward according to the condition
        if truncated:
            reward = Reward.FOOD.value
            self._snake.grow()
            self.score += 1
            self._goal = self._place_food()
        elif terminated:
            reward = Reward.COLLISION.value
        else:
            reward = Reward.COLLISION_FREE.value
            # reward = np.exp(-np.linalg.norm(Line(self.snake.head.center, self._food.center).to_vector() / self.pixel_size))
        return self.observations, reward, terminated, self.info

    def render(self, canvas=None):
        if (self.render_mode == "human") and (self._screen is None):
            self._init_human_renderer()
        if canvas is None:
            canvas = pygame.Surface(self.window_size)
            canvas.fill(Colors.BLACK.value)

        self.draw(canvas)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self._screen.blit(canvas, canvas.get_rect())
            # Draw the text showing the score
            text = self._font.render(f"Score: {self.score}", True, Colors.WHITE.value)
            self._screen.blit(text, [0, 0])
            # update the display
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self._clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.render_mode == "human":
            self._screen = None
            self._font = None
            self._clock = None
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed : Optional[int] = None):
        if seed is None:
            seed = self._seed
        assert isinstance(seed, int), "Only integers are allowed for seeding"
        self.rng = np.random.default_rng(seed)

    def draw(self, canvas: pygame.Surface):
        super().draw(canvas)
        self._snake.draw(canvas)

    def check_overlaps(self):
        """Check overlaps between the snake, the food and the obstacles.

        Raises:
            CollisionError: error raised if one of the snake body part or food collide with the obstacles in the environment or with themself
        """
        # Check collisions for the snake
        for snake_part in self._snake:
            if snake_part.colliderect(self._food):
                raise CollisionError(f"The snake part {snake_part} collides with the food {self._food}.")
            collision_idx = snake_part.collidelist(self._obstacles)
            if collision_idx != -1:
                raise CollisionError(f"The snake part {snake_part} collides with the obstacle {self._obstacles[collision_idx]}.")
        # Check collisions for the food
        food_collision = self._goal.collidelist(self._obstacles)
        if food_collision != -1:
            raise CollisionError(f"The food {self._goal} collides with the obstacle {self._obstacles[food_collision]}.")

    ## Properties
    @property
    def free_positions(self):
        "Available positions from the free position mask"
        if self._free_position_mask is None:
            raise InitialisationError("The free position mask is not initialised. Reset the environment first !")
        return [(x, y) for x in range(self.width) for y in range(self.height) if self._free_position_mask[x, y]]

    @property
    def snake(self):
        "Snake agent represented by a subclass of Snake."
        if self._snake is None:
            raise InitialisationError("The snake is not initialised. Reset the environment first !")
        return self._snake

    @property
    def observations(self):
        "Observation associated with the current state of the environment"
        if self._position is None:
            raise InitialisationError("The position is not initialised. Reset the environment first !")
        left, front, right = self._get_neighbours()

        return np.array([
            ## Neighbours collision
            self._is_collision(left), # LEFT
            self._is_collision(front), # FRONT
            self._is_collision(right), # RIGHT
            ## Snake direction
            self._snake.direction == Direction.UP, # UP
            self._snake.direction == Direction.RIGHT, # RIGHT
            self._snake.direction == Direction.DOWN, # DOWN
            self._snake.direction == Direction.LEFT, # LEFT
            ## Food position
            self._goal.y < self._position.y, # UP
            self._goal.x > self._position.x, # RIGHT
            self._goal.y > self._position.y, # DOWN
            self._goal.x < self._position.x, # LEFT
        ], dtype=int)

    @property
    def info(self):
        "Informations about the snake game states."
        if self._position is None:
            raise InitialisationError("The position is not initialised. Reset the environment first !")
        return {
            "snake_direction": self._snake.direction,
            "obstacles": self._obstacles,
            "snake_head": self._position,
            "food": self._goal,
            "truncated": self._position.colliderect(self._goal),
        }

    ## Private methods
    def _init_human_renderer(self):
        pygame.init()
        # instanciation of arguments that will be used by pygame when drawing
        self._screen = pygame.display.set_mode(self.window_size)
        self._font = pygame.font.SysFont("freesansbold.ttf", 30)
        self._clock = pygame.time.Clock()

    #TODO : Think about a new way to initialize snake !
    def _place_snake(self) -> SnakeAI:
        # As the snake is initialised along
        # available_positions = [(x, y) for x in range(2, self.width) for y in range(self.height) if all(self._free_positions[x-2:x+1, y])]
        # assert len(available_positions) > 0, "There is no available positions for the snake in the current environment"
        available_positions = self.free_positions
        self.rng.shuffle(available_positions)
        snake_positions = self._get_snake_positions(available_positions)
        # x, y = self.rng.choice(self.free_positions)
        # snake = SnakeAI(x * self.pixel, y * self.pixel, pixel=self.pixel)
        snake = SnakeAI(snake_positions, pixel=self.pixel)
        # self._free_position_mask[x-2:x+1, y-2:y+1] = False
        return snake

    def _get_snake_positions(self, available_positions : List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        for x, y in available_positions:
            # RIGHT
            if (x +1 , y) in available_positions:
                if (x + 1, y - 1) in available_positions:
                    return [(x, y), (x + 1, y), (x + 1, y - 1)]
                if (x + 2 , y) in available_positions:
                    return [(x, y), (x + 1, y), (x + 2, y)]
                if (x + 1, y + 1) in available_positions:
                    return [(x, y), (x + 1, y), (x + 1, y + 1)]
            # BOTTOM
            if (x , y + 1) in available_positions:
                if (x + 1 , y + 1) in available_positions:
                    return [(x, y), (x, y + 1), (x + 1, y + 1)]
                if (x, y + 2) in available_positions:
                    return [(x, y), (x, y + 1), (x, y + 2)]
                if (x - 1, y + 1) in available_positions:
                    return [(x, y), (x, y + 1), (x - 1, y + 1)]
            # LEFT
            if (x - 1, y) in available_positions:
                if (x - 1 , y + 1) in available_positions:
                    return [(x, y), (x - 1, y), (x - 1, y + 1)]
                if (x - 2, y) in available_positions:
                    return [(x, y), (x - 1, y), (x - 2, y)]
                if (x - 1, y - 1) in available_positions:
                    return [(x, y), (x - 1, y), (x - 1, y - 1)]
            # TOP
            if (x , y - 1) in available_positions:
                if (x - 1 , y - 1) in available_positions:
                    return [(x, y), (x, y - 1), (x - 1, y - 1)]
                if (x, y - 2) in available_positions:
                    return [(x, y), (x, y - 1), (x, y - 2)]
                if (x + 1, y - 1) in available_positions:
                    return [(x, y), (x, y - 1), (x + 1, y - 1)]
        raise ConfigurationError("There is no valid configuration in free space to place a 3 pixel snake.")

    def _place_food(self) -> Rectangle:
        # Define the central coordinates of the food to be placed
        # assert len(available_positions) > 0, "There is no available positions for the food in the current environment"
        x, y = self.rng.choice(self.free_positions)
        food = Rectangle(x * self.pixel, y * self.pixel, self.pixel, self.pixel)
        self._free_position_mask[x,y] = False
        return food

    def _populate_grid_with_obstacles(self) -> List[Rectangle]:
        obstacles = []
        # Loop over the obstacle sizes
        for i, nb_obstacle in enumerate(self._nb_samples_per_size[::-1]):
            size = self._max_obs_size - i
            for _ in range(nb_obstacle):
                obstacles.append(self._place_obstacle(size))
        return obstacles

    def _place_obstacle(self, size: int) -> Rectangle:
        assert size > 0, f"Obstacle size need to be at least 1. Get {size}"
        # available_positions = [(x, y) for x in range(self.width-(size-1)) for y in range(self.height-(size-1)) if self._free_positions[x, y]]
        # assert len(available_positions) > 0, f"There is no available position for an obstacle of size {size}"
        x, y = self.rng.choice(self.free_positions)
        obstacle = Rectangle(x * self.pixel, y * self.pixel, size * self.pixel, size * self.pixel)
        # Remove all possible
        if size > 1:
            self._free_position_mask[x:x+(size-1), y:y+(size-1)] = False
        else:
            self._free_position_mask[x, y] = False
        return obstacle

    # Collision handling
    def _is_outside(self, rect: Optional[pygame.Rect] = None) -> bool:
        if rect is None:
            rect = self._snake.head
        return rect.x < 0 or rect.x + rect.width > self.window_size[0] or rect.y < 0 or rect.y + rect.height > self.window_size[1]

    def _collide_with_obstacles(self, rect: Optional[pygame.Rect]  = None) -> bool:
        if rect is None:
            return self._snake.collide_with_obstacles(self._obstacles)
        return rect.collidelist(self._obstacles) != -1

    def _collide_with_snake_body(self, rect: Optional[pygame.Rect] = None) -> bool:
        if rect is None:
            return self._snake.collide_with_itself()
        return rect.collidelist(self._snake.body) != -1

    def _is_collision(self, rect: Optional[pygame.Rect] = None) -> bool:
        return self._is_outside(rect) or self._collide_with_snake_body(rect) or self._collide_with_obstacles(rect)

    def _get_neighbours(self) -> List[pygame.Rect]:
        """Return left, front and right neighbouring bounding boxes located at snake head

        Raises:
            ValueError: if the snake direction is not in [UP, RIGHT, DOWN, LEFT]

        Returns:
            _type_: _description_
        """
        bottom = self._position.move(0, self.pixel)
        top = self._position.move(0, -self.pixel)
        left = self._position.move(-self.pixel, 0)
        right = self._position.move(self.pixel, 0)

        if self._snake.direction == Direction.UP:
            return left, top, right
        if self._snake.direction == Direction.RIGHT:
            return top, right, bottom
        if self._snake.direction == Direction.DOWN:
            return right, bottom, left
        if self._snake.direction == Direction.LEFT:
            return bottom, left, top
        raise ValueError(f'Unknown direction {self._snake.direction}')

    ## Dunder methods
    def __eq__(self, other: object) -> bool:
        assert isinstance(other, SnakeEnv), f"Can not compare equality with an instance of {type(other)}. Expected type is SnakeEnv"
        size_check = (self.height == other.height) and (self.width == other.width) and (self.pixel == other.pixel)
        snake_check = self.snake == other.snake
        food_check = self.goal == other.goal
        obstacles_check = self.obstacles == other.obstacles
        return size_check and snake_check and food_check and obstacles_check

    def __repr__(self) -> str:
        return f"{__class__.__name__}(width={self.width!r}, height={self.height!r}, pixel={self.pixel!r}, nb_obstacles={self.nb_obs!r}, max_obs_size={self._max_obs_size!r}, render_mode={self.render_mode!r}, seed={self._seed})"

if __name__ == "__main__":
    import time
    snake_env = SnakeEnv(20, 20, nb_obstacles=10, max_obs_size=10, render_mode="human")
    snake_env.reset()
    fps = 10

    action = 0
    done = False
    while not done:
        # time.sleep(1/fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                snake_env.close()
                quit()
            key_pressed = event.type == pygame.KEYDOWN
            if key_pressed and event.key == pygame.K_UP:
                action = 0
            if key_pressed and event.key == pygame.K_RIGHT:
                action = 1
            # if key_pressed and event.key == pygame.K_DOWN:
            #     action = 2
            if key_pressed and event.key == pygame.K_LEFT:
                action = 2
        _, _, terminated, _ = snake_env.step(action)
        if terminated:
            snake_env.reset()
            print('You suck ! Try again !')
        snake_env.render()