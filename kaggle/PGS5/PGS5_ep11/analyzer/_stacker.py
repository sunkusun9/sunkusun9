import re
import numpy as np
from modeler._data_wrapper import DataWrapper

class Stacker:
    dataset = 'dataset'

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

    def _set_node(self, idx, node):
        build_id = self.build_ids.get((node, idx), '')
        current_build_id = ''.join([i['build_id'] for _, _, i in self.e.nodes[node].objs_[idx]])

        if build_id == current_build_id and (node, idx) in self.result:
            return

        self.build_ids[(node, idx)] = current_build_id

        iterator = self.e.get_node_valid_output(idx, node, self.output_var)
        aggregated = self._aggregate(iterator)

        self.result[(node, idx)] = aggregated.to_array()
        self.columns[(node, idx)] = aggregated.get_columns()

    def set_node(self, node):
        for idx in range(self.e.get_n_splits()):
            self._set_node(idx, node)

    def set_nodes(self, query):
        if isinstance(query, str):
            nodes = self.e.get_node_names(query)
        elif isinstance(query, re.Pattern):
            nodes = self.e.get_node_names(query)
        elif isinstance(query, list):
            nodes = query
        else:
            raise ValueError(f"query must be str, re.Pattern or list, got {type(query)}")

        for idx in range(self.e.get_n_splits()):
            for node in nodes:
                self._set_node(idx, node)

    def unset_node(self, node):
        keys_to_remove = [k for k in self.result.keys() if k[0] == node]
        for k in keys_to_remove:
            del self.result[k]
            del self.columns[k]
            del self.build_ids[k]

    def get_dataset(self):
        nodes = list(set(k[0] for k in self.result.keys()))

        if not nodes:
            return None

        wrapper_cls = type(self.target_value_)
        wrappers = []

        if self.include_target:
            wrappers.append(self.target_value_)

        for node in nodes:
            data_list = []
            columns = None
            for idx in range(self.e.get_n_splits()):
                if (node, idx) in self.result:
                    data_list.append(self.result[(node, idx)])
                    if columns is None:
                        columns = self.columns[(node, idx)]

            combined_arr = np.concatenate(data_list, axis=0)
            node_wrapper = wrapper_cls.from_output(
                combined_arr,
                column_names=columns,
                index=self.target_value_.get_index()
            )
            wrappers.append(node_wrapper)

        combined = wrapper_cls.concat(wrappers, axis=1)
        return combined.iloc(self.sort_order_)
