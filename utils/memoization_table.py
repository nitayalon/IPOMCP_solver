from abc import ABC
from IPOMCP_solver.Solver.ipomcp_config import *
import pandas as pd
import numpy as np


class MemoizationTable(ABC):
    def __init__(self, path_to_memoization_dir):
        self.path_to_dir = path_to_memoization_dir
        self.config = get_config()
        self.device = self.config.device
        self.original_data = self.load_data()
        self.data = self.original_data
        self.new_data = pd.DataFrame()

    def load_data(self):
        return pd.DataFrame()

    def save_data(self):
        pass

    @staticmethod
    def load_results(directory_name):
        pass

    def query_table(self, query_parameters: dict):
        pass

    def update_table(self, q_values: np.array, history: np.array, beliefs: np.array, game_parameters: dict):
        pass
