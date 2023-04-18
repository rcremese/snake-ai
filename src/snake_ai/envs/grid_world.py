##
# @author  <robin.cremese@gmail.com>
 # @file Base class for all 2D path planning environments with discret action space
 # @desc Created on 2023-03-13 1:08:45 pm
 # @copyright MIT License
 #
from abc import ABCMeta, abstractmethod
from snake_ai.envs import Rectangle
from snake_ai.envs.agent import Agent
from snake_ai.envs.walker import Walker2D
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

    def __init__(self, width : int = 20, height : int = 20, pixel : int = 10, seed: int = 0, render_mode : Optional[str]=None, **kwargs) -> None:
        """
        Args:
            width (int, optional): Environment width in terms of `metapixel`. Defaults to 20.
            height (int, optional): Environment height in terms of `metapixel`. Defaults to 20.
            pixel (int, optional): Size of the environment `metapixel` in terms of classical pixel. Defaults to 10.
            seed (int, optional): Seed for the Random Number Generator (RNG) of the environment (numpy.random). Defaults to 0.
            render_mode (str, optional): Name of the mode the environment should be rendered. If None, there is no rendering. Defaults to None.
        Raises:
            TypeError: Raised if the inputs are not instance of ints.
            ValueError: Raised if the inputs are negative ints or the render mode is not one of the None,
            ValueError: Raised if the render mode is not one of the following : None, "human", "rgb_array"
        """
        # Initialise all numerical parameters
        if any(param <=0 for param in [width, height, pixel]):
            raise ValueError("Only positive integers are allowed for (width, height, pixel). " +
                f"Get ({width},{height}, {pixel})")
        self.width, self.height, self.pixel  = int(width), int(height), int(pixel)
        self.window_size = (self.width * self.pixel, self.height * self.pixel)
        self._free_position_mask = np.ones((self.width, self.height))

        # Initialise the render mode
        if (render_mode is not None) and (render_mode not in self.metadata["render_modes"]):
            raise ValueError(f"render_mode should be eather None or in {self.metadata['render_modes']}. Get {render_mode}")
        self.render_mode = render_mode

        # Gym env attributes
        self.observation_space = gym.spaces.MultiBinary(12)
        self.action_space = gym.spaces.Discrete(4)

        # All non-instanciated attributes
        self._screen = None
        self._goal = None
        self._agent = None
        self._obstacles = None
        self._truncated = None

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
        self._truncated = False
        self._free_position_mask = np.ones((self.width, self.height))
        # Initialise obstacles
        self._obstacles = []
        # Initialise goal & position
        x_goal, y_goal = self._rng.choice(self.free_positions)
        self.goal = Rectangle(x_goal * self.pixel, y_goal * self.pixel, self.pixel, self.pixel)
        x_agent, y_agent = self._rng.choice(self.free_positions)
        self.agent = Walker2D(x_agent, y_agent, self.pixel)
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
        self.agent.move_from_action(action)
        # A flag is set if the agent has reached the goal
        self._truncated = self.agent.position.colliderect(self.goal)
        # An other one if the snake is outside or collide with obstacles
        terminated = self._is_collision()
        # Give a reward according to the condition
        if self._truncated:
            reward = Reward.FOOD.value
            self.score += 1
            x_goal, y_goal = self._rng.choice(self.free_positions)
            self.goal = Rectangle(x_goal * self.pixel, y_goal * self.pixel, self.pixel, self.pixel)
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
        # pygame.draw.rect(canvas, Colors.BLUE1.value, self.position)
        self.agent.draw(canvas)
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
        assert rect.width == rect.height == self.pixel, f"Only squares of size {self.pixel} are allowed to represent the goal."
        self._sanity_check(rect)
        # Remove the old position to the free position mask
        if self._goal is not None:
            x, y = self._get_grid_position(self._goal)
            self._free_position_mask[x, y] = True
        # Update free position map
        x, y = self._get_grid_position(rect)
        self._free_position_mask[x, y] = False
        self._goal = rect

    @property
    def agent(self) -> Agent:
        """Current position of the agent, represented by a rectangle"""
        if self._agent is None:
            raise InitialisationError("The position variable is not initialized. Reset the environment !")
        return self._agent

    @agent.setter
    def agent(self, agent : Agent):
        self._sanity_check(agent.position)
        self._agent = agent

    # @property
    # def direction(self) -> Direction:
    #     """Current direction of the agent"""
    #     if self._direction is None:
    #         raise InitialisationError("The direction argument is not initialized. Reset the environment !")

    # @direction.setter
    # def direction(self, direction : Direction):
    #     if direction not in Direction:
    #         raise ValueError(f"Unknown direction {direction}. Expected one of the following : {Direction}")
    #     self._direction = direction

    @property
    def obstacles(self) -> List[Rectangle]:
        """Obstacles in the environment, represented by a list of rectangle"""
        if self._obstacles is None:
            raise InitialisationError("The obstacles are not initialized. Reset the environment !")
        return self._obstacles



    @property
    def observations(self) -> np.ndarray:
        """Observation associated with the current state of the environment

        Binary vector representing :
        - the collisions of agent neighbours with obstacles and walls
        - the current direction of the agent
        - the goal position relative to the agent.
        """
        north, east, south, west= self.agent.neighbours

        return np.array([
            ## Neighbours collision
            self._is_collision(north), # TOP
            self._is_collision(east), # RIGHT
            self._is_collision(south), # BOTTOM
            self._is_collision(west), # LEFT
            ## Snake direction
            self.agent.direction == Direction.NORTH, # UP
            self.agent.direction == Direction.EAST, # RIGHT
            self.agent.direction == Direction.SOUTH, # DOWN
            self.agent.direction == Direction.WEST, # LEFT
            ## Food position
            self.goal.y < self.agent.position.y, # UP
            self.goal.x > self.agent.position.x, # RIGHT
            self.goal.y > self.agent.position.y, # DOWN
            self.goal.x < self.agent.position.x, # LEFT
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
        return {
            "agent_position": self.agent.position,
            "agent_direction": self.agent.direction,
            "obstacles": self._obstacles,
            "goal": self._goal,
            "truncated": self._truncated,
        }


    @property
    def free_positions(self) -> List[Tuple[int, int]]:
        """Available free positions represented as a list of position tuple (x, y) taken from the free position mask"""
        if self._free_position_mask is None:
            raise InitialisationError("The free position mask is not initialised. Reset the environment first !")
        return np.argwhere(self._free_position_mask).tolist()

    ## Private methods
    def _sanity_check(self, rect : Rectangle):
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
        # TODO : Add the possibility to add rectangles as obstacles
        if rect.height != rect.width:
            raise ShapeError(f"Only squares are accepted in the environment. Get (width, height) = ({rect.width}, {rect.height}).")
        if (rect.x < 0) or (rect.x + rect.width > self.window_size[0]) or (rect.y < 0) or (rect.y + rect.height > self.window_size[1]):
            raise OutOfBoundsError(f"The rectangle position ({rect.x}, {rect.y}) is out of bounds {self.window_size}")
        if any([corner % self.pixel != 0 for corner in [rect.right, rect.top, rect.left, rect.bottom]]):
            raise ResolutionError(f"The rectangle positions and lengths need to be a factor of pixel size : {self.pixel}.")

    # Initialisation methods
    def _init_human_renderer(self):
        """Initialisation method of the human rendered based on pygame.
        """
        pygame.init()
        # instanciation of arguments that will be used by pygame when drawing
        self._screen = pygame.display.set_mode(self.window_size)
        self._font = pygame.font.SysFont("freesansbold.ttf", 30)
        self._clock = pygame.time.Clock()

    def _get_grid_position(self, rect : Rectangle) -> Tuple[int, int]:
        """Return the grid position of a rectangle

        Args:
            rect (Rectangle): Rectangle to be converted

        Returns:
            Tuple[int, int]: Grid position of the rectangle
        """
        return (rect.x // self.pixel, rect.y // self.pixel)

    # Collision handling
    def _is_outside(self, rect: Optional[pygame.Rect] = None) -> bool:
        """Check wether the input rectangle or the agent is outside of the environment bounds.

        Args:
            rect (pygame.Rect, optional): Rectangle to check. If None, the agent position is used. Defaults to None.

        Returns:
            bool: Flag that indicate if the input rectangle or the agent is outside
        """
        if rect is None:
            rect = self.agent.position
        return rect.x < 0 or rect.x + rect.width > self.window_size[0] or rect.y < 0 or rect.y + rect.height > self.window_size[1]

    def _collide_with_obstacles(self, rect: Optional[pygame.Rect]  = None) -> bool:
        """Check wether the input rectangle or the agent collides with obstacles in the environment.

        Args:
            rect (pygame.Rect, optional): Rectangle to check. If None, the agent position is used. Defaults to None.

        Returns:
            bool: Flag that indicate if the input rectangle or the agent collides with obstacles
        """
        if rect is None:
            rect = self.agent.position
        return rect.collidelist(self._obstacles) != -1

    def _is_collision(self, rect: Optional[pygame.Rect] = None) -> bool:
        return self._is_outside(rect) or self._collide_with_obstacles(rect)

    ## Dunder methods
    def __eq__(self, other: object) -> bool:
        assert isinstance(other, GridWorld), f"Can not compare equality with an instance of {type(other)}. Expected type is GridWorld"
        size_check = (self.height == other.height) and (self.width == other.width) and (self.pixel == other.pixel)
        position_check = self.agent == other.agent
        goal_check = self.goal == other.goal
        obstacles_check = self.obstacles == other.obstacles
        return size_check and position_check and goal_check and obstacles_check

    def __repr__(self) -> str:
        return f"{__class__.__name__}(width={self.width}, height={self.height}, pixel={self.pixel}, render_mode={self.render_mode}, seed={self._seed})"

if __name__ == "__main__":
    snake_env = GridWorld(20, 20, render_mode="human")
    seed = 0
    snake_env.reset(seed)

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
            print('You suck ! Try again !')
        snake_env.render()