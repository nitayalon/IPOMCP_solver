import os
from typing import Union

import confuse
import torch


class Config(object):
    def __init__(self, environment, args, path: str):
        self._config = confuse.Configuration('Solver', __name__)
        self._config.set_file(path)
        self.env = environment
        self.args = args
        self.game_params = None
        self.cuda_is_available = torch.cuda.is_available()
        torch.manual_seed(self.seed)
        self.device = torch.device("cuda" if self.cuda_is_available else "cpu")
        self.q_values_dir, self.simulation_results_dir = self.create_experiment_dir()

    def create_experiment_dir(self):
        path_prefix = self.get_from_general("results_folder")
        worker_agent = self.args.worker_tom  # self.get_from_env("agents")['worker']['agent_name']
        manager_agent = self.args.manager_tom  # self.get_from_env("agents")['manager']['agent_name']
        first_mover = self.args.first_mover
        experiment_name = f'{manager_agent}_manager_{worker_agent}_worker_{first_mover}_begins_softmax_{self.args.softmax_temp}'
        q_values_dir = os.path.join(str(path_prefix), self.env, experiment_name, 'q_values')
        simulation_results_dir = os.path.join(str(path_prefix), self.env, experiment_name, 'simulation_results')
        os.makedirs(q_values_dir, exist_ok=True)
        os.makedirs(simulation_results_dir, exist_ok=True)
        return q_values_dir, simulation_results_dir

    def get_agent_tom_level(self, role):
        if role == "manager":
            return self.manager_tom_level
        else:
            return self.worker_tom_level

    def get_from_env(self, key_in_env=None):
        res = self._config['environments'][f'{self.env}']
        if key_in_env:
            res = res[key_in_env]
        return res.get()

    def get_from_env_exploration(self, key_in_env=None):
        res = self._config['environments'][f'{self.env}']['exploration']
        if key_in_env:
            res = res[key_in_env]
        return res.get()

    def get_from_general(self, key_in_general=None):
        res = self._config['general']
        if key_in_general:
            res = res[key_in_general]
        return res.get()

    def get_from_models(self, key_in_models=None):
        res = self._config['models']
        if key_in_models:
            res = res[key_in_models]
        return res.get()

    def get_from_game_env(self, game_env_key: str = None):
        if game_env_key is not None:
            return self.game_params.get(game_env_key)
        return self.game_params

    def get_from_env_agents(self, role, key_from_agent: str = None):
        if key_from_agent:
            return self._config['environments'][self.env]['agents'][role][key_from_agent].get()
        return self._config['environments'][self.env]['agents'][role].get()

    def get_agent(self, name: str):
        return self._config['agents'][f'{name}'].get()

    def get_key_path(self, *path_keys):
        cur_val = None
        for k in path_keys:
            if not cur_val:
                cur_val = self._config[k]
            else:
                cur_val = cur_val[k]

        return cur_val

    def set_game_env(self, game_id, game_params):
        self.game_params = {
            "game_id": game_id,
            "budget": game_params[0],
            "labor_costs": game_params[1],
            "fee": game_params[2]
        }

    @property
    def seed(self):
        return self.args.seed

    @property
    def softmax_temperature(self):
        return self.args.softmax_temp

    @property
    def first_acting_agent(self):
        return self.args.first_mover

    @property
    def manager_tom_level(self):
        return self.args.manager_tom

    @property
    def worker_tom_level(self):
        return self.args.worker_tom

_config = None


def init_config(environment, args, path: str = 'config.yaml') -> Config:
    global _config
    if not _config:
        _config = Config(environment, args, path)

    return _config


def get_config() -> Union[Config, None]:
    if _config is None:
        raise RuntimeError("config was accessed before setting it using init_config(env)")
    return _config
