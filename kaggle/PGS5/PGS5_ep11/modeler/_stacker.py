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

    def _build_sort_order(self):
        all_valid_idx = np.concatenate([
            self.experimenter.valid_idx_list[i] for i in range(self.experimenter.get_n_splits())
        ])
        return np.argsort(all_valid_idx)

    def _build_target_value(self):
        target_list = []
        for idx in range(self.experimenter.get_n_splits()):
            iterator = self.experimenter.get_data_valid(idx, self.target_edges)
            aggregated = DataWrapper.simple(iterator)
            target_list.append(aggregated)

        wrapper_cls = type(target_list[0])
        return wrapper_cls.concat(target_list, axis=0)

    def _aggregate(self, iterator):
        if self.method == 'simple':
            return DataWrapper.simple(iterator)
        elif self.method == 'mean':
            return DataWrapper.mean(iterator)
        elif self.method == 'mode':
            return DataWrapper.mode(iterator)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _get_valid(self, result_data):
        return result_data['output_valid'].select_columns(
            resolve_columns(result_data['output_valid'], self.output_var)
        )

    def _start(self, node):
        self.stacking[node] = list()

    def _stack(self, node, idx, stk):
        l = self.stacking[node]
        if len(l) != idx:
            raise RuntimeError("")
        self.stacking[node].append(
            stk.to_array()
        )
        if idx == 0:
            self.columns[node] = stk.get_columns()
    
    def _end(self, node):
        self.stacking[node] = np.concatenate(self.stacking[node])
    
    def _get_nodes(self, nodes):
        if nodes is None:
            # 기존 동작: 모든 root group의 노드
            node_names = list(self.stacking.keys())
        elif isinstance(nodes, list):
            node_names = [n for n in nodes if n in self.stacking]
        elif isinstance(nodes, str):
            pat = re.compile(nodes)
            node_names = [k for k in self.stacking.keys() if k is not None and pat.search(k)]
        else:
            raise ValueError(f"nodes must be None, list, or str, got {type(nodes)}")
        return node_names

    def reset_nodes(self, nodes):
        for node in self._get_nodes(nodes):
            del self.metrics[node]
    
    def get_dataset(self, nodes):
        node_names = self._get_nodes(nodes)
        target = self._build_target_value()
        wrapper_cls = type(target)
        column_names = list()
        for i in node_names:
            column_names.extend(self.columns[i])

        return wrapper_cls.concat([
            wrapper_cls.from_output(
                np.concatenate(
                    [self.stacking[i] for i in node_names], axis = 1
                ), column_names, target.get_index()
            )
        ] + [target], axis=1).iloc(self._build_sort_order()).data