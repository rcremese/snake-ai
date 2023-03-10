from abc import ABCMeta, abstractmethod
from snake_ai.envs import Rectangle
import pygame

from snake_ai.utils.errors import InitialisationError, ResolutionError, OutOfBoundsError, ShapeError
from snake_ai.utils import Colors
from typing import List

class GridWorld(metaclass=ABCMeta):
    """Abstract base class for 2D path planning environments with discret action space.
    """
    def __init__(self, width : int, height : int, pixel : int, nb_obs : int = 0, max_obs_size : int = 1) -> None:
        """
        Args:
            width (int): Environment width in terms of `metapixel.
            height (int): Environment height in terms of `metapixel.`
            pixel (int): Size of the environment `metapixel` in terms of classical pixel.
            nb_obs (int, optional): Number of obstacles in the environment. Defaults to 0.
            max_obs_size (int, optional): Maximum obstacle size in terms of `metapixel`. Defaults to 1.

        Raises:
            TypeError: Raised if the inputs are not instance of ints.
            ValueError: Raised if the inputs are negative ints.
        """
        if not all([isinstance(param, int) for param in [width, height, pixel, nb_obs, max_obs_size]]):
            raise TypeError("Only positive integers are allowed for (width, height, pixel, nb_obs).")
        if any([param <=0 for param in [width, height, pixel, max_obs_size]]):
            raise ValueError("Only positive integers are allowed for (width, height, pixel, max_obs_size). " +
                f"Get ({width},{height}, {pixel}, {max_obs_size})")
        self.width, self.height, self.pixel  = width, height, pixel
        self.window_size = (self.width * self.pixel, self.height * self.pixel)
        self.nb_obs, self._max_obs_size = nb_obs, max_obs_size

        self._obstacles = None
        self._goal = None
        self._position = None
        self._free_position_mask = None

    ## Public methods
    @abstractmethod
    def reset():
        raise NotImplementedError()

    def draw(self, canvas : pygame.Surface):
        assert isinstance(canvas, pygame.Surface), f"Canvas must be an instance of pygame.Surface, not {type(canvas)}."
        # Draw agent position, obstacles and goal
        pygame.draw.rect(canvas, Colors.BLUE1.value, self.position)
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(canvas, Colors.RED.value, obstacle)
        # Draw goal
        pygame.draw.rect(canvas, Colors.GREEN.value, self.goal)

    # Properties
    @property
    def goal(self) -> Rectangle:
        "Goal to reach"
        if self._goal is None:
            raise InitialisationError("The goal variable is not initialized. Reset the environment !")
        return self._goal

    @goal.setter
    def goal(self, rect : Rectangle):
        self._sanity_check(rect, self.pixel)
        self._goal = rect

    @property
    def position(self) -> Rectangle:
        "Current position"
        if self._position is None:
            raise InitialisationError("The position variable is not initialized. Reset the environment !")
        return self._position

    @position.setter
    def position(self, rect : Rectangle):
        self._sanity_check(rect, self.pixel)
        self._position = rect

    @property
    def obstacles(self) -> List[Rectangle]:
        "Obstacles in the environment"
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

    ## Private methods
    def _sanity_check(self, rect : Rectangle, max_size : int):
        """Check the validity of a rectangle in the GridWorld environment.

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
        if (rect.x < 0) or (rect.x > self.window_size[0]) or (rect.y < 0) or (rect.y > self.window_size[1]):
            raise OutOfBoundsError(f"The rectangle position ({rect.x}, {rect.y}) is out of bounds {self.window_size}")
        if any([corner % self.pixel != 0 for corner in [rect.x, rect.y, rect.width, rect.height]]):
            raise ResolutionError(f"The rectangle positions and lengths need to be a factor of pixel size : {self.pixel}.")
        if rect.height > max_size or rect.height > max_size:
            raise ShapeError(f"The rectangle length can not be greater than {max_size}. " +
                              f"Get (width, height) = {rect.width}, {rect.height}.")

    def __repr__(self) -> str:
        return f"{__class__.__name__}(width={self.width}, height={self.height}, pixel={self.pixel}, nb_obs={self.nb_obs}, max_obs_size={self._max_obs_size})"
