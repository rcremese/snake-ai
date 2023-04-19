##
# @author  <robin.cremese@gmail.com>
 # @file Description
 # @desc Created on 2023-04-19 12:44:32 pm
 # @copyright MIT License
 #
from snake_ai.envs import GridWorld, RandomObstaclesEnv, RoomEscape, MazeGrid, SnakeEnv, SlotEnv
from snake_ai.physim.simulation import Simulation, DiffusionSimulation

from abc import ABCMeta, abstractmethod
from typing import Union, Any, Dict
from pathlib import Path
from phi import flow
import json
class Writter(metaclass=ABCMeta):
    def __init__(self, path : Union[Path, str]):
        self.path = Path(path).resolve()

    @abstractmethod
    def write(self, data : Any):
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def convert_to_dict(data : Any):
        raise NotImplementedError()
class EnvWritter(Writter):
    def write(self, env : GridWorld):
        dictionary = self.convert_to_dict(env)
        with open(self.path, 'w') as f:
            json.dump(dictionary, f)

    @staticmethod
    def convert_to_dict(env : GridWorld) -> Dict[str, Any]:
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

class SimulationWritter(Writter):
    def __init__(self, path : Union[Path, str]):
        super().__init__(path)

    def write(self, simulation : Simulation):
        if not self.path.exists():
            self.path.mkdir(parents=True)

        dictionary = self.convert_to_dict(simulation)
        with open(self.path.joinpath("simulation.json"), 'w') as f:
            json.dump(dictionary, f)
        flow.field.write(simulation.field, str(self.path.joinpath("field")))


    @staticmethod
    def convert_to_dict(simulation : Simulation) -> Dict[str, Any]:
        dictionary = {
            "name": simulation.__class__.__name__,
            "parameters": simulation.hparams,
            "env": EnvWritter.convert_to_dict(simulation.env),
        }
        return dictionary
class Loader(metaclass=ABCMeta):
    def __init__(self, path : Union[Path, str]):
        self.path = Path(path).resolve(strict=True)

    @abstractmethod
    def load(self) -> Any:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def load_from_dict(self, dictionary : Dict[str, Any]) -> Any:
        raise NotImplementedError()
class EnvLoader(Loader):
    def load(self) -> GridWorld:
        with open(self.path, 'r') as file:
            dictionary = json.load(file)
        return self.load_from_dict(dictionary)

    @staticmethod
    def load_from_dict(dictionary : dict) -> GridWorld:
        keys = {'name', 'render_mode', 'width', 'height', 'pixel', 'seed'}
        assert keys.issubset(dictionary.keys()), f"One of the following keys is not in the input dictionary : {keys}"

        if dictionary['render_mode'] == 'None':
            dictionary['render_mode'] = None

        if dictionary['name'] == 'GridWorld':
            return GridWorld(**dictionary)
        elif dictionary['name'] == 'RandomObstaclesEnv':
            keys = {'nb_obstacles', 'max_obs_size'}
            assert keys.issubset(dictionary.keys()), f"One of the following keys is not in the input dictionary : {keys}"
            return RandomObstaclesEnv(**dictionary)
        elif dictionary['name'] == 'SnakeEnv':
            keys = {'nb_obstacles', 'max_obs_size', 'snake_type'}
            assert keys.issubset(dictionary.keys()), f"One of the following keys is not in the input dictionary : {keys}"
            return SnakeEnv(**dictionary)
        elif dictionary['name'] == 'MazeGrid':
            assert 'maze_generator' in dictionary.keys(), f"One of the following keys is not in the input dictionary : maze_generator"
            return MazeGrid(**dictionary)
        elif dictionary['name'] == 'RoomEscape':
            return RoomEscape(**dictionary)
        elif dictionary['name'] == 'SlotEnv':
            return SlotEnv(**dictionary)
        else:
            raise NotImplementedError(f"Environment {dictionary['name']} is not implemented")

class SimulationLoader(Loader):
    def __init__(self, path : Union[Path, str]):
        super().__init__(path)
        assert self.path.is_dir(), f"Path {self.path} is not a directory"
        assert self.path.joinpath("simulation.json").exists(), f"Path {self.path} does not contain a simulation.json file"
        assert self.path.joinpath("field.npz").exists(), f"Path {self.path} does not contain a field.npz file"

    def load(self) -> Simulation:
        with open(self.path.joinpath("simulation.json"), 'r') as file:
            dictionary = json.load(file)
        simulation = self.load_from_dict(dictionary)
        simulation.reset()
        simulation.field = flow.field.read(str(self.path.joinpath("field.npz")))
        return simulation

    @staticmethod
    def load_from_dict(dictionary : dict) -> Simulation:
        keys = {'name', 'parameters', 'env'}
        assert keys.issubset(dictionary.keys()), f"One of the following keys is not in the input dictionary : {keys}"

        if dictionary['name'] == 'DiffusionSimulation':
            env = EnvLoader.load_from_dict(dictionary['env'])
            return DiffusionSimulation(env=env, **dictionary['parameters'])
        else:
            raise NotImplementedError(f"Simulation {dictionary['name']} is not implemented")