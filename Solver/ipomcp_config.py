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
        self.task_duration = self.get_from_env("n_trials")
        self.game_params = None
        self.environment_name = None
        self.experiment_name = None
        self.subintentional_agent_type = str(self.get_from_env("subintentional_type"))
        self.planning_results_dir, self.simulation_results_dir, self.beliefs_dir, self.q_values_results_dir, \
        self.path_to_memoization_data = self.create_experiment_dir()
        self.cuda_is_available = torch.cuda.is_available()
        torch.manual_seed(self.seed)
        self.device = torch.device("cuda" if self.cuda_is_available else "cpu")
        self.report_ipocmp_statistics = bool(self.get_from_general("report_ipocmp_statistics"))
        self.output_planning_tree = bool(self.get_from_general("output_planning_tree"))

    def create_experiment_dir(self):
        path_prefix = self.get_from_general("results_folder")
        duration = self.get_from_env("n_trials")
        duration = "long_duration" if duration == 20 else "short_duration"
        sender_tom = self.args.sender_tom
        receiver_tom = self.args.receiver_tom
        environment_name = f'{receiver_tom}_receiver_{sender_tom}_sender_softmax_temp_{self.args.softmax_temp}'
        which_senders = self._infer_senders_types()
        self.environment_name = f'{environment_name}_{which_senders}'
        experiment_path = os.path.join(str(path_prefix), self.env, duration)
        general_path = os.path.join(experiment_path, f'{environment_name}_{which_senders}')
        # Export MCTS trees
        planning_results_dir = os.path.join(str(general_path), 'planning_results')
        # Export q_values
        q_values_results_dir = os.path.join(str(general_path), 'q_values')
        # Export game outcomes
        simulation_results_dir = os.path.join(str(general_path), 'simulation_results')
        # Export beliefs
        beliefs_dir = os.path.join(str(general_path), 'beliefs')
        # Memoization data
        memoization_dir = os.path.join(experiment_path, 'memoization')
        os.makedirs(planning_results_dir, exist_ok=True)
        os.makedirs(simulation_results_dir, exist_ok=True)
        os.makedirs(beliefs_dir, exist_ok=True)
        os.makedirs(q_values_results_dir, exist_ok=True)
        return planning_results_dir, simulation_results_dir, beliefs_dir, q_values_results_dir, memoization_dir

    def get_agent_tom_level(self, role):
        if role == "rational_sender":
            return self.args.sender_tom
        else:
            return self.args.receiver_tom

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
    def agent_tom_level(self):
        return self.args.agent_tom

    @property
    def subject_tom_level(self):
        return self.args.subject_tom

    def new_experiment_name(self, experiment_name):
        self.experiment_name = experiment_name

    def _infer_senders_types(self):
        random_agent = True
        rational_agent = True
        if self.get_from_general("skip_random"):
            random_agent = False
        if self.get_from_general("skip_rational"):
            rational_agent = False
        return f'random_sender_included_{random_agent}_rational_sender_included_{rational_agent}'


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
