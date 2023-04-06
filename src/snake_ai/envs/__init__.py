from .snake_classic_env import SnakeClassicEnv
from .snake_lines_env import SnakeLinesEnv
from .snake import Snake, BidirectionalSnake, SnakeHuman
from .walker import Walker2D
from .geometry import Rectangle, Circle
from .grid_world import GridWorld
from .random_obstacles_env import RandomObstaclesEnv
from .room_escape import RoomEscape
from .maze_grid import MazeGrid
from .snake_env import SnakeEnv
from gym.envs.registration import register

register(
    id='Snake-v0',
    entry_point='snake_ai.envs.snake_classic_env:SnakeClassicEnv',
    kwargs={'render_mode' : None, 'width' : 20, 'height': 20, 'nb_obstacles': 0, 'pixel' : 10, 'max_obs_size' : 3}
)

register(
    id='SnakeLines-v0',
    entry_point='snake_ai.env.snake_lines_env:SnakeLinesEnv',
    kwargs={'render_mode' : None, 'width' : 20, 'height': 20, 'nb_obstacles': 0, 'pixel' : 10, 'max_obs_size' : 3}
)