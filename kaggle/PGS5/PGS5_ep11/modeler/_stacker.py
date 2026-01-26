import re
import numpy as np
from modeler._data_wrapper import DataWrapper
from ._node_processor import resolve_columns

class Stacker:
    def __init__(self, experimenter, target_edges, output_var, method='mean', include_target=True):
        self.experimenter = experimenter
        self.target_edges = target_edges
        self.output_var = output_var
        self.method = method
        self.include_target = include_target
        self.columns = {}
        self.stacking = {}

        # self._build_target_value()
        # self._build_sort_order()

    def build_sort_order(self):
        all_valid_idx = np.concatenate([
            self.experimenter.valid_idx_list[i] for i in range(self.experimenter.get_n_splits())
        ])
        return np.argsort(all_valid_idx)

    def build_target_value(self):
        target_list = []
        for idx in range(self.experimenter.get_n_splits()):
            iterator = self.experimenter.get_node_valid_output(idx, self.target_edge[0], self.target_edge[1])
            aggregated = DataWrapper.simple(iterator)
            target_list.append(aggregated)

        wrapper_cls = type(target_list[0])
        return wrapper_cls.concat(target_list, axis=0)

    def aggregate(self, iterator):
        if self.method == 'simple':
            return DataWrapper.simple(iterator)
        elif self.method == 'mean':
            return DataWrapper.mean(iterator)
        elif self.method == 'mode':
            return DataWrapper.mode(iterator)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def get_valid(self, result_data):
        return result_data['output_valid'].select_columns(
            resolve_columns(result_data['output_valid'], self.output_var)
        )
    