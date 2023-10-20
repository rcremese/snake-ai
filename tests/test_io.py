from snake_ai.utils.io import SimulationWritter, SimulationLoader, EnvWriter, EnvLoader
from snake_ai.physim import DiffusionSimulation
from snake_ai.envs import GridWorld
from pathlib import Path
import json


class TestEnvIO:
    env = GridWorld(10, 10, 10)
    env.reset()

    # TODO : Test for all envs
    def test_write(self, tmp_path: Path):
        filepath = tmp_path.joinpath("env.json")
        writter = EnvWriter(filepath)
        writter.write(self.env)
        assert filepath.exists()
        with open(filepath, "r") as f:
            env_dict = json.load(f)
        assert env_dict["name"] == "GridWorld"
        assert env_dict["width"] == 10
        assert env_dict["height"] == 10
        assert env_dict["pixel"] == 10
        assert env_dict["seed"] == 0
        assert env_dict["render_mode"] == "None"

    def test_load(self, tmp_path: Path):
        filepath = tmp_path.joinpath("env.json")
        writter = EnvWriter(filepath)
        writter.write(self.env)
        loader = EnvLoader(filepath)
        env = loader.load()
        env.reset()
        assert env == self.env


class TestSimulationIO:
    env = GridWorld(10, 10, 10)
    simulation = DiffusionSimulation(env, t_max=10, dt=0.1)
    simulation.reset()

    def test_write(self, tmp_path: Path):
        writter = SimulationWritter(tmp_path)
        writter.write(self.simulation)
        assert tmp_path.joinpath("simulation.json").exists()
        assert tmp_path.joinpath("field.npz").exists()

    def test_load(self, tmp_path: Path):
        writter = SimulationWritter(tmp_path)
        writter.write(self.simulation)
        loader = SimulationLoader(tmp_path)
        simulation = loader.load()
        assert simulation.t_max == self.simulation.t_max
        assert simulation.dt == self.simulation.dt
        assert simulation.env == self.simulation.env
        assert simulation.field == self.simulation.field
