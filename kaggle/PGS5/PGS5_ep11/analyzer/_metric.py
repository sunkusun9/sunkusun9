import re
import pandas as pd

class Metric:
    metric = 'metric'

    def __init__(
        self, e, target_edge, output_var, metric_func, include_train = False
    ):
        self.e = e
        self.target_edge = target_edge
        self.output_var = output_var
        self.include_train = include_train
        self.metric_func = metric_func
        self.result = {}
        self.build_ids = {}

    def _set_node(self, idx, node):
        build_id = self.build_ids.get((node, idx), '')
        current_build_id = ''.join([i['build_id'] for _, _, i in self.e.nodes[node].objs_[idx]])

        if build_id == current_build_id and (node, idx) in self.result:
            return

        self.build_ids[(node, idx)] = current_build_id

        if self.include_train:
            iterator = zip(
                self.e.get_node_output(idx, self.target_edge[0], self.target_edge[1]),
                self.e.get_node_output(idx, node, self.output_var)
            )
            for no, ((true_train, true_valid), (prd_train, prd_valid)) in enumerate(iterator):
                result_train = self.metric_func(true_train[0].data, prd_train[0].data)
                result_valid = self.metric_func(true_valid.data, prd_valid.data)
                if true_train[1] is not None:
                    result_sub = {
                        (idx, 'train', f'train_{no}'): result_train,
                        (idx, 'train', f'valid_{no}'): self.metric_func(true_train[1].data, prd_train[1].data),
                        (idx, 'valid', ''): result_valid
                    }
                else:
                    result_sub = {
                        (idx, 'train'): result_train, (idx, 'valid'): result_valid
                    }
        else:
            iterator = zip(
                self.e.get_node_valid_output(idx, self.target_edge[0], self.target_edge[1]),
                self.e.get_node_valid_output(idx, node, self.output_var)
            )
            for true_valid, prd_valid in iterator:
                result_valid = self.metric_func(true_valid.data, prd_valid.data)
                result_sub = {idx: result_valid}

        self.result[(node, idx)] = pd.Series(result_sub)

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
            del self.build_ids[k]

    def get_metric(self, selected_nodes=None):
        if selected_nodes is None:
            nodes = list(set(k[0] for k in self.result.keys()))
        elif isinstance(selected_nodes, str):
            nodes = self.e.get_node_names(selected_nodes)
        elif isinstance(selected_nodes, re.Pattern):
            nodes = self.e.get_node_names(selected_nodes)
        elif isinstance(selected_nodes, list):
            nodes = selected_nodes
        else:
            raise ValueError(f"selected_nodes must be str, re.Pattern, list or None, got {type(selected_nodes)}")

        nodes = [n for n in nodes if any(k[0] == n for k in self.result.keys())]

        if not nodes:
            return None

        grps = {node: self.e.get_parents(node) for node in nodes}

        c, mx = None, -1
        for i in grps.values():
            i = i[::-1]
            if c is None:
                c = i
            elif len(c) > 0:
                mx = max(mx, len(i))
                for j in range(min(len(c), len(i))):
                    if c[j] != i[j]:
                        c = i[:j]
                        break
            else:
                break

        result = {}
        for node in nodes:
            node_series = pd.concat([
                self.result[(node, idx)] for idx in range(self.e.get_n_splits())
                if (node, idx) in self.result
            ])
            if c and len(c) > 0:
                i = tuple([''] * (mx - len(grps[node]) - len(c)) + [node])
            else:
                i = tuple([''] * (mx - len(grps[node])) + [node])
            result[node] = node_series.rename(i)

        return pd.DataFrame(result.values())
