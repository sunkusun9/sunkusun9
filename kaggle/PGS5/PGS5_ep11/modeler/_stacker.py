import re
import numpy as np
from modeler._data_wrapper import DataWrapper

class Stacker:
    def __init__(self, e, target_edge, output_var, method='mean', include_target=True):
        self.e = e
        self.target_edge = target_edge
        self.output_var = output_var
        self.method = method
        self.include_target = include_target
        self.result = {}
        self.columns = {}
        self.build_ids = {}

        self._build_target_value()
        self._build_sort_order()

    def _build_sort_order(self):
        all_valid_idx = np.concatenate([
            self.e.valid_idx_list[i] for i in range(self.e.get_n_splits())
        ])
        self.sort_order_ = np.argsort(all_valid_idx)

    def _build_target_value(self):
        target_list = []
        for idx in range(self.e.get_n_splits()):
            iterator = self.e.get_node_valid_output(idx, self.target_edge[0], self.target_edge[1])
            aggregated = DataWrapper.simple(iterator)
            target_list.append(aggregated)

        wrapper_cls = type(target_list[0])
        self.target_value_ = wrapper_cls.concat(target_list, axis=0)

    def _aggregate(self, iterator):
        if self.method == 'simple':
            return DataWrapper.simple(iterator)
        elif self.method == 'mean':
            return DataWrapper.mean(iterator)
        elif self.method == 'mode':
            first = next(iterator)
            wrapper_cls = type(first)
            def new_iterator():
                yield first
                for i in iterator:
                    yield i
            return wrapper_cls.mode(new_iterator())
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def get_stack(self, idx, node):
        build_id = self.build_ids.get((node, idx), '')
        current_build_id = ''.join([i['build_id'] for _, _, i in self.e.nodes[node].objs_[idx]])

        if build_id == current_build_id and (node, idx) in self.result:
            return

        self.build_ids[(node, idx)] = current_build_id

        iterator = self.e.get_node_valid_output(idx, node, self.output_var)
        aggregated = self._aggregate(iterator)

        return (aggregated.to_array(), aggregated.get_columns())

    