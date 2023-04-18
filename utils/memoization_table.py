from abc import ABC
from IPOMCP_solver.Solver.ipomcp_config import *
import pandas as pd
import numpy as np


class MemoizationTable(ABC):
    def __init__(self):
        self.config = get_config()
        self.device = self.config.device
        self.data = self.load_behavioural_data()

    def load_behavioural_data(self):
        return pd.DataFrame()

    @staticmethod
    def load_results(directory_name):
        pass

    def query_table(self, query_parameters: dict):
        pass

    def update_table(self, q_values: np.array, history: np.array, beliefs: np.array, game_parameters: dict):
        pass
