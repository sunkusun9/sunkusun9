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

    def _get_data(self, idx):
        return list(
            self.experimenter.get_data(idx, self.target_edges)
        )

    def _get_metric(self, target_data, result_data):
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

    def _start(self, node):
        self.metrics[node] = list()

    def _set_metric(self, node, idx, metric):
        l = self.metrics[node]
        if len(l) != idx:
            raise RuntimeError("")
        l.append(metric)

    def _end(self, node):
        pass

    def _get_nodes(self, nodes):
        if nodes is None:
            # 기존 동작: 모든 root group의 노드
            node_names = list(self.metrics.keys())
        elif isinstance(nodes, list):
            node_names = [n for n in nodes if n in self.metrics]
        elif isinstance(nodes, str):
            pat = re.compile(nodes)
            node_names = [k for k in self.metrics.keys() if k is not None and pat.search(k)]
        else:
            raise ValueError(f"nodes must be None, list, or str, got {type(nodes)}")
        return node_names
    
    def reset_nodes(self, nodes):
        for node in self._get_nodes(nodes):
            del self.metrics[node]

    def get_metric(self, node):
        l = list()
        for i, sub in enumerate(self.metrics[node]):
            l.append(
                pd.concat([pd.Series(j, name = str(no)) for no, j in enumerate(sub)], axis=1).unstack()
            )
        return pd.concat(l, axis=1).unstack(level=[0, 1]).rename(node)

    def get_metrics(self, nodes):
        node_names = self._get_nodes(nodes)
        return pd.concat([self.get_metric(node) for node in node_names], axis=1).T