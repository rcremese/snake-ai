from .snake_classic_env import SnakeClassicEnv
from .snake_lines_env import SnakeLinesEnv
from .grid_world import GridWorld
from .snake import SnakeAI, SnakeHuman
from .geometry import Rectangle, Circle

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