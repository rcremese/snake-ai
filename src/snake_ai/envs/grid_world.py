##
# @author  <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2023-03-13 1:08:45 pm
 # @copyright MIT License
 #
from snake_ai.envs.geometry import Rectangle
import pygame

import numpy as np
import gym.spaces
import gym

from snake_ai.utils.errors import InitialisationError, ResolutionError, OutOfBoundsError, ShapeError, ConfigurationError
from snake_ai.utils import Colors, Direction, Reward
from typing import List, Optional, Tuple, Dict, Any

class GridWorld(gym.Env):
    """Base class for all 2D path planning environments with discret action space.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, width : int = 20, height : int = 20, pixel : int = 10, nb_obs : int = 0,
                 max_obs_size : int = 1, seed: int = 0, render_mode : Optional[str]=None):
        """
        Args:
            width (int, optional): Environment width in terms of `metapixel`. Defaults to 20.
            height (int, optional): Environment height in terms of `metapixel`. Defaults to 20.
            pixel (int, optional): Size of the environment `metapixel` in terms of classical pixel. Defaults to 10.
            nb_obs (int, optional): Number of obstacles in the environment. Defaults to 0.
            max_obs_size (int, optional): Maximum obstacle size in terms of `metapixel`. Defaults to 1.
            seed (int, optional): Seed for the Random Number Generator (RNG) of the environment (numpy.random). Defaults to 0.
            render_mode (str, optional): Name of the mode the environment should be rendered. If None, there is no rendering. Defaults to None.
        Raises:
            TypeError: Raised if the inputs are not instance of ints.
            ValueError: Raised if the inputs are negative ints or the render mode is not one of the None,
            ValueError: Raised if the render mode is not one of the following : None, "human", "rgb_array"
        """
        # Initialise all numerical parameters
        if not all([isinstance(param, int) for param in [width, height, pixel, nb_obs, max_obs_size]]):
            raise TypeError("Only positive integers are allowed for (width, height, pixel, nb_obs).")
        if any([param <=0 for param in [width, height, pixel, max_obs_size]]):
            raise ValueError("Only positive integers are allowed for (width, height, pixel, max_obs_size). " +
                f"Get ({width},{height}, {pixel}, {max_obs_size})")
        self.width, self.height, self.pixel  = width, height, pixel
        self.window_size = (self.width * self.pixel, self.height * self.pixel)
        self.nb_obs, self._max_obs_size = nb_obs, max_obs_size
        # Initialise the render mode
        if (render_mode is not None) and (render_mode not in self.metadata["render_modes"]):
            raise ValueError(f"render_mode should be eather None or in {self.metadata['render_modes']}. Get {render_mode}")
        self.render_mode = render_mode

        # Gym env attributes
        self.observation_space = gym.spaces.MultiBinary(12)
        self.action_space = gym.spaces.Discrete(4)

        # All non-instanciated attributes
        self._obstacles = None
        self._screen = None
        self._goal = None
        self._position = None
        self._direction = None
        self._free_position_mask = None

        if self.render_mode == "human":
            self._init_human_renderer()

        self.seed(seed)

    ## Public methods
    def reset(self, seed : Optional[int]=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Method to reset the environment

        Args:
            seed (Optional[int], optional): Seed to be used for environment generation. If None, use the previously defined seed, otherwise overwrite the seed value. Defaults to None.

        Returns:
            observations, info: Vector of 0 and 1 that represent the observations and dictionary of available informations
        """
        self.seed(seed)
        # Initialise a grid of free positions and score
        self.score = 0
        # self.truncated = False
        self._free_position_mask = np.ones((self.width, self.height))
        # Initialise obstacles
        self._obstacles = []
        if self.nb_obs > 0:
            self._obstacles = self._populate_grid_with_obstacles()
        # Initialise goal & position
        self._goal = self._place_goal()
        self._position = self._place_goal(mask_position=False)
        self._truncated = False
        self._direction = Direction.NORTH
        return self.observations, self.info

    def step(self, action : int) -> Tuple[np.ndarray, Reward, bool, Dict]:
        """Method to take a step forward in the environment given the input action

        Args:
            action (int): Integer in the range [0,3] that represent the direction in which to move. Respectively [UP, RIGHT, DOWN, LEFT]

        Raises:
            ValueError: Raised if the action is not in the range [0,3]

        Returns:
            Tuple[np.ndarray, Reward, bool, Dict]: Classical "observations, reward, done, info" signals sent by Gym environments
        """
        if action == 0:
            self._direction = Direction.NORTH
            self._position.move_ip(0, -self.pixel)
        elif action == 1:
            self._direction = Direction.EAST
            self._position.move_ip(self.pixel, 0)
        elif action == 2:
            self._direction = Direction.SOUTH
            self._position.move_ip(0, self.pixel)
        elif action == 3:
            self._direction = Direction.WEST
            self._position.move_ip(-self.pixel, 0)
        else:
            raise ValueError(f"Expected action values are [0, 1, 2, 3]. Get {action}")

        # A flag is set if the agent has reached the goal
        self._truncated = self._position.colliderect(self._goal)
        # An other one if the snake is outside or collide with obstacles
        terminated = self._is_collision()
        # Give a reward according to the condition
        if self._truncated:
            reward = Reward.FOOD.value
            self.score += 1
            self._goal = self._place_goal()
        elif terminated:
            reward = Reward.COLLISION.value
        else:
            reward = Reward.COLLISION_FREE.value
        return self.observations, reward, terminated, self.info

    def render(self, canvas : pygame.Surface=None) -> Optional[np.ndarray]:
        """Render the environment with respect to the "render_mode" argument

        Args:
            canvas (pygame.Surface, optional): Surface on wich to render the environment. If None, initialise a new surface. Defaults to None.

        Returns:
            np.ndarray (optional): a numpy array representation of the RGB image rendered by the method. Only returned if "render_mode"=="rgb_array"
        """
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

    def draw(self, canvas : pygame.Surface):
        """Draw obstacles, agent and goal on canvas.

        Args:
            canvas (pygame.Surface): Surface on which to draw.
        """
        assert isinstance(canvas, pygame.Surface), f"Canvas must be an instance of pygame.Surface, not {type(canvas)}."
        # Draw agent position, obstacles and goal
        pygame.draw.rect(canvas, Colors.BLUE1.value, self.position)
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(canvas, Colors.RED.value, obstacle)
        # Draw goal
        pygame.draw.rect(canvas, Colors.GREEN.value, self.goal)

    def close(self):
        """Close the pygame display opened if render_mode is "human"
        """
        if self._screen is not None:
            pygame.display.quit()
            self._screen = None
            pygame.quit()

    def seed(self, seed : Optional[int] = None):
        """Method used to seed the environment

        Args:
            seed (int, optional): Seed to use for the RNG. If None, use the previous environment seed, otherwise overwrite its value. Defaults to None.
        """
        if seed is not None:
            assert isinstance(seed, int), "Only integers are allowed for seeding"
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

    ## Properties
    @property
    def goal(self) -> Rectangle:
        """Goal to reach, represented by a rectangle"""
        if self._goal is None:
            raise InitialisationError("The goal variable is not initialized. Reset the environment !")
        return self._goal

    @goal.setter
    def goal(self, rect : Rectangle):
        self._sanity_check(rect, self.pixel)
        self._goal = rect

    @property
    def position(self) -> Rectangle:
        """Current position of the agent, represented by a rectangle"""
        if self._position is None:
            raise InitialisationError("The position variable is not initialized. Reset the environment !")
        return self._position

    @position.setter
    def position(self, rect : Rectangle):
        self._sanity_check(rect, self.pixel)
        self._position = rect

    @property
    def direction(self) -> Direction:
        """Current direction of the agent"""
        if self._direction is None:
            raise InitialisationError("The direction argument is not initialized. Reset the environment !")

    @direction.setter
    def direction(self, direction : Direction):
        if direction not in Direction:
            raise ValueError(f"Unknown direction {direction}. Expected one of the following : {Direction}")
        self._direction = direction

    @property
    def obstacles(self) -> List[Rectangle]:
        """Obstacles in the environment, represented by a list of rectangle"""
        if self._obstacles is None:
            raise InitialisationError("The obstacles are not initialized. Reset the environment !")
        return self._obstacles

    @obstacles.setter
    def obstacles(self, rectangles : List[Rectangle]):
        # Simple case in which the user provide only 1 rectangle
        if isinstance(rectangles, Rectangle):
            rectangles = [rectangles]
        for rect in rectangles:
            self._sanity_check(rect, self._max_obs_size * self.pixel)
        self._obstacles = rectangles

    @property
    def observations(self) -> np.ndarray:
        """Observation associated with the current state of the environment

        Binary vector representing :
        - the collisions of agent neighbours with obstacles and walls
        - the current direction of the agent
        - the goal position relative to the agent.
        """
        if self._position is None:
            raise InitialisationError("The position is not initialised. Reset the environment first !")
        up, right, down, left= self._get_neighbours()

        return np.array([
            ## Neighbours collision
            self._is_collision(up), # TOP
            self._is_collision(right), # RIGHT
            self._is_collision(down), # BOTTOM
            self._is_collision(left), # LEFT
            ## Snake direction
            self._direction == Direction.NORTH, # UP
            self._direction == Direction.EAST, # RIGHT
            self._direction == Direction.SOUTH, # DOWN
            self._direction == Direction.WEST, # LEFT
            ## Food position
            self._goal.y < self._position.y, # UP
            self._goal.x > self._position.x, # RIGHT
            self._goal.y > self._position.y, # DOWN
            self._goal.x < self._position.x, # LEFT
        ], dtype=int)

    @property
    def info(self) -> Dict[str, Any]:
        """Informations about the environment states arranged as a dictionnary.

        Contains:
        - current direction of the agent (instance of [Direction](snake_ai.utils.direction.Direction))
        - obstacles in the environment (list of [Rectangle](snake_ai.envs.geometry.Rectangle))
        - current position of the agent (instance of [Rectangle](snake_ai.envs.geometry.Rectangle))
        - current position of the goal (instance of [Rectangle](snake_ai.envs.geometry.Rectangle))
        - boolean flag that indicates if the agent reached the goal at the current step
        """
        if self._position is None:
            raise InitialisationError("The position is not initialised. Reset the environment first !")
        return {
            "direction": self._direction,
            "obstacles": self._obstacles,
            "position": self._position,
            "goal": self._goal,
            "truncated": self._truncated,
        }

    @property
    def free_positions(self) -> List[Tuple[int, int]]:
        """Available free positions represented as a list of position tuple (x, y) taken from the free position mask"""
        if self._free_position_mask is None:
            raise InitialisationError("The free position mask is not initialised. Reset the environment first !")
        return [(x, y) for x in range(self.width) for y in range(self.height) if self._free_position_mask[x, y]]

    ## Private methods
    def _sanity_check(self, rect : Rectangle, max_size : int):
        """Check the validity of a rectangle position in the GridWorld environment.

        Args:
            rect (Rectangle): Rectangle to be checked
            max_size (int): Maximum size of a rectangle width and height

        Raises:
            TypeError: Raised if the input is not an istance of Rectangle
            OutOfBoundsError: Raised if the input rectangle is out of bounds
            ResolutionError: Raised if the rectnagle position or length is not a multiple of environment pixel size
            ShapeError: Raised if the rectangle length is greater than the maximum size.
        """
        if not isinstance(rect, Rectangle):
            raise TypeError("Only rectangles are allowed in GridWorld.")
        if self._is_outside(rect):
            raise OutOfBoundsError(f"The rectangle {rect} is out of bounds {self.window_size}")
        if any([corner % self.pixel != 0 for corner in [rect.top, rect.left, rect.bottom, rect.right]]):
            raise ResolutionError(f"The rectangle positions and lengths need to be a factor of pixel size : {self.pixel}.")
        if rect.height > max_size or rect.height > max_size:
            raise ShapeError(f"The rectangle length can not be greater than {max_size}. " +
                              f"Get (width, height) = {rect.width}, {rect.height}.")

    # Initialisation methods
    def _init_human_renderer(self):
        """Initialisation method of the human rendered based on pygame.
        """
        pygame.init()
        # instanciation of arguments that will be used by pygame when drawing
        self._screen = pygame.display.set_mode(self.window_size)
        self._font = pygame.font.SysFont("freesansbold.ttf", 30)
        self._clock = pygame.time.Clock()

    def _place_goal(self, mask_position : bool = True) -> Rectangle:
        """Return a rectangle randomly located on the free positions grid and update the free_position_mask

        Args:
            mask_position (bool, optional): Flag that indicate if the sampled rectangle position should be masked. Used for agent placement. Defaults to True.

        Returns:
            Rectangle: A rectangle that can represent either the agent or the goal.
        """
        # Define the central coordinates of the food to be placed
        # assert len(available_positions) > 0, "There is no available positions for the food in the current environment"
        x, y = self._rng.choice(self.free_positions)
        goal = Rectangle(x * self.pixel, y * self.pixel, self.pixel, self.pixel)
        if mask_position:
            self._free_position_mask[x,y] = False
        return goal

    def _populate_grid_with_obstacles(self) -> List[Rectangle]:
        """Populate the environment with obstacles of various sizes.

        Raises:
            ConfigurationError: If the total area occupied by of the obstacles is greater than the total area of the environment.

        Returns:
            List[Rectangle]: Randomly spaced obstacles represented as a list of rectangles.
        """
        obstacles = []
        # Construct a list with the number of obstacle samples per size (first value = 1 metapixel).
        nb_samples_per_size = self._max_obs_size * [self.nb_obs // self._max_obs_size, ]
        # If the number of obstacles can not be devided by the maximum value, samples are added to the obstacles of size 1
        nb_samples_per_size[0] += self.nb_obs - sum(nb_samples_per_size)
        total_area = sum([(size + 1) * nb_sample**2 for size, nb_sample in enumerate(nb_samples_per_size)])
        if total_area >= self.width * self.height - 2:
            raise ConfigurationError(f"There are too much obstacles ({self.nb_obs}) or with a range which is too wide {self._max_obs_size} for the environment ({self.width},{self.height}).")
        nb_samples_per_size.reverse()
        # Loop over the obstacle sizes
        for i, nb_obstacle in enumerate(nb_samples_per_size):
            size = self._max_obs_size - i
            for _ in range(nb_obstacle):
                obstacles.append(self._place_obstacle(size))
        return obstacles

    # TODO : cleaner le code
    def _place_obstacle(self, size: int) -> Rectangle:
        """Place an obstacle of a given size in the environment while repecting the free position condition

        Args:
            size (int): size of the obstacle

        Returns:
            Rectangle: Obstacle to place in the environment, represented as a square of size 1.
        """
        assert size > 0, f"Obstacle size need to be at least 1. Get {size}"
        # available_positions = [(x, y) for x in range(self.width-(size-1)) for y in range(self.height-(size-1)) if self._free_positions[x, y]]
        # assert len(available_positions) > 0, f"There is no available position for an obstacle of size {size}"
        x, y = self._rng.choice(self.free_positions)
        obstacle = Rectangle(x * self.pixel, y * self.pixel, size * self.pixel, size * self.pixel)
        # Remove all possible
        self._free_position_mask[x:x+size, y:y+size] = False
        # if size > 1:
        #     self._free_position_mask[x:x+size, y:y+size] = False
        # else:
        #     self._free_position_mask[x, y] = False
        return obstacle

    def _get_neighbours(self) -> List[pygame.Rect]:
        """Return left, front and right neighbouring bounding boxes located at snake head

        Raises:
            ValueError: if the snake direction is not in [UP, RIGHT, DOWN, LEFT]

        Returns:
            List[pygame.Rect]: clockwise neighbour recangles starting at positions top
        """
        down = self._position.move(0, self.pixel)
        up = self._position.move(0, -self.pixel)
        left = self._position.move(-self.pixel, 0)
        right = self._position.move(self.pixel, 0)
        return up, right, down, left

    # Collision handling
    def _is_outside(self, rect: Optional[pygame.Rect] = None) -> bool:
        """Check wether the input rectangle or the agent is outside of the environment bounds.

        Args:
            rect (pygame.Rect, optional): Rectangle to check. If None, the agent position is used. Defaults to None.

        Returns:
            bool: Flag that indicate if the input rectangle or the agent is outside
        """
        if rect is None:
            rect = self._position
        return rect.x < 0 or rect.x + rect.width > self.window_size[0] or rect.y < 0 or rect.y + rect.height > self.window_size[1]

    def _collide_with_obstacles(self, rect: Optional[pygame.Rect]  = None) -> bool:
        """Check wether the input rectangle or the agent collides with obstacles in the environment.

        Args:
            rect (pygame.Rect, optional): Rectangle to check. If None, the agent position is used. Defaults to None.

        Returns:
            bool: Flag that indicate if the input rectangle or the agent collides with obstacles
        """
        if rect is None:
            rect = self._position
        return rect.collidelist(self._obstacles) != -1

    def _is_collision(self, rect: Optional[pygame.Rect] = None) -> bool:
        return self._is_outside(rect) or self._collide_with_obstacles(rect)

    ## Dunder methods
    def __eq__(self, other: object) -> bool:
        assert isinstance(other, GridWorld), f"Can not compare equality with an instance of {type(other)}. Expected type is GridWorld"
        size_check = (self.height == other.height) and (self.width == other.width) and (self.pixel == other.pixel)
        position_check = self.position == other.position
        goal_check = self.goal == other.goal
        obstacles_check = self.obstacles == other.obstacles
        return size_check and position_check and goal_check and obstacles_check

    def __repr__(self) -> str:
        return f"{__class__.__name__}(width={self.width}, height={self.height}, pixel={self.pixel}, nb_obs={self.nb_obs}, max_obs_size={self._max_obs_size}, render_mode={self.render_mode}, seed={self._seed})"

if __name__ == "__main__":
    import time
    snake_env = GridWorld(20, 20, nb_obs=10, max_obs_size=10, render_mode="human")
    seed = 0
    snake_env.reset(seed)
    print(snake_env._free_position_mask)
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
            if key_pressed and event.key == pygame.K_DOWN:
                action = 2
            if key_pressed and event.key == pygame.K_LEFT:
                action = 3
        _, _, terminated, _ = snake_env.step(action)
        if terminated:
            seed += 1
            snake_env.reset(seed)
            print(snake_env._free_position_mask)
            print('You suck ! Try again !')
        snake_env.render()