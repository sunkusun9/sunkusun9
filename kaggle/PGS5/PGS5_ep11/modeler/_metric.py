import re
import pandas as pd
from ._node_processor import resolve_columns

class Metric:
    def __init__(
        self, name, experimenter, target_edges, output_var, metric_func, include_train = False
    ):
        self.experimenter = experimenter
        self.name = name
        self.target_edges = target_edges
        self.output_var = output_var
        self.include_train = include_train
        self.metric_func = metric_func
        self.metrics = dict()

    def get_data(self, idx):
        return list(
            self.experimenter.get_data(idx, self.target_edges)
        )

    def get_metric(self, target_data, result_data):
        (true_train_t, true_train_v), true_valid = target_data
        selected_cols = resolve_columns(result_data['output_valid'], self.output_var)
        prd = result_data['output_valid'].select_columns(selected_cols)
        result = {
            'valid': self.metric_func(true_valid.data, prd.data)
        }
        if self.include_train:
            prd = result_data['output_train'][0].select_columns(selected_cols)
            result['train_sub'] = self.metric_func(true_train_t.data, prd.data)
            if true_train_v is not None:
                prd = result_data['output_train'][1].select_columns(selected_cols)
                result['valid_sub'] = self.metric_func(true_train_v.data, prd.data)
        return result