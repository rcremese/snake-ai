from snake_ai.envs import GridWorld, RandomObstaclesEnv, RoomEscape, MazeGrid, SnakeEnv

from abc import ABCMeta, abstractmethod
from typing import Union, Any
from pathlib import Path
import json 
class Writter(metaclass=ABCMeta):
    def __init__(self, path : Union[Path, str]):
        self.path = Path(path).resolve()

    @abstractmethod
    def write(self, data : Any):
        raise NotImplementedError()

    @abstractmethod
    def convert_to_dict(self, data : Any):
        raise NotImplementedError()
class EnvWritter(Writter):
    def write(self, env : GridWorld):
        dictionary = self.convert_to_dict(env)
        with open(self.path, 'w') as f:
            json.dump(dictionary, f)

    def convert_to_dict(self, env : GridWorld):
        dictionary = {
            "name": env.__class__.__name__,
            "width": env.width,
            "height": env.height,
            "pixel": env.pixel,
            "seed": env._seed,
            "render_mode": "None" if env.render_mode is None else env.render_mode,
        }
        if isinstance(env, RandomObstaclesEnv):
            dictionary["nb_obstacles"] = env._nb_obs
            dictionary["max_obs_size"] = env._max_obs_size
        if isinstance(env, SnakeEnv):
            dictionary["snake_type"] = env._snake_type
        if isinstance(env, MazeGrid):
            dictionary["maze_generator"] = env.maze_generator
        return dictionary
class Loader(metaclass=ABCMeta):
    def __init__(self, path : Union[Path, str]):
        self.path = Path(path).resolve(strict=True)
        
    @abstractmethod
    def load(self) -> Any:
        raise NotImplementedError()
    
    @abstractmethod
    def load_from_dict(self, dictionary : dict) -> Any:
        raise NotImplementedError()
class EnvLoader(Loader):
    def load(self) -> GridWorld:
        with open(self.path, 'r') as file:
            dictionary = json.load(file)
        self.load_from_dict(dictionary)

    def load_from_dict(self, dictionary : dict) -> GridWorld:
        keys = {'name', 'render_mode', 'width', 'height', 'pixel', 'seed'}
        assert keys.issubset(dictionary.keys()), f"One of the following keys is not in the input dictionary {self.path.name} : {keys}"
        
        if dictionary['name'] == 'GridWorld':
            return GridWorld(**dictionary)
        elif dictionary['name'] == 'RandomObstaclesEnv':
            keys = {'nb_obstacles', 'max_obs_size'}
            assert keys.issubset(dictionary.keys()), f"One of the following keys is not in the input dictionary {self.path.name} : {keys}"
            return RandomObstaclesEnv(**dictionary)
        elif dictionary['name'] == 'SnakeEnv':
            keys = {'nb_obstacles', 'max_obs_size', 'snake_type'}
            assert keys.issubset(dictionary.keys()), f"One of the following keys is not in the input dictionary {self.path.name} : {keys}"
            return SnakeEnv(**dictionary)
        elif dictionary['name'] == 'MazeGrid':
            assert 'maze_generator' in dictionary.keys(), f"One of the following keys is not in the input dictionary {self.path.name} : maze_generator"
            return MazeGrid(**dictionary)
        elif dictionary['name'] == 'RoomEscape':
            return RoomEscape(**dictionary)
        else:
            raise NotImplementedError(f"Environment {dictionary['name']} is not implemented")
