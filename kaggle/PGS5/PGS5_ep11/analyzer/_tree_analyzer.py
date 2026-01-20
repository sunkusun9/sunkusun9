import re
import pandas as pd

class DecisionTreeAnalyzer:
    feature_importances = 'feature_importances'
    tree = 'tree'

    def __init__(self, e):
        self.e = e
        self.result = {}
        self.build_ids = {}

    def _set_node(self, idx, node):
        build_id = self.build_ids.get((node, idx), '')
        current_build_id = ''.join([i['build_id'] for _, _, i in self.e.nodes[node].objs_[idx]])

        if build_id == current_build_id and (node, idx) in self.result:
            return

        self.build_ids[(node, idx)] = current_build_id

        inner_results = {}
        for inner_idx, (processor, _, _) in enumerate(self.e.nodes[node].objs_[idx]):
            input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else []

            feature_importances = pd.Series(
                processor.feature_importances_,
                index=input_vars,
                name='feature_importances'
            )

            tree_ = processor.tree_
            tree_structure = []
            for i in range(tree_.node_count):
                feature_idx = tree_.feature[i]
                node_dict = {
                    'node_id': i,
                    'feature': input_vars[feature_idx] if feature_idx >= 0 else None,
                    'threshold': tree_.threshold[i] if feature_idx >= 0 else None,
                    'impurity': tree_.impurity[i],
                    'n_samples': tree_.n_node_samples[i],
                    'left_child': tree_.children_left[i] if tree_.children_left[i] >= 0 else None,
                    'right_child': tree_.children_right[i] if tree_.children_right[i] >= 0 else None,
                    'value': tree_.value[i].tolist()
                }
                tree_structure.append(node_dict)

            inner_results[inner_idx] = {
                'feature_importances': feature_importances,
                'tree': tree_structure
            }

        self.result[(node, idx)] = inner_results

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

    def get_feature_importances(self, node):
        if (node, 0) not in self.result:
            self.set_node(node)

        dfs = list()
        for i in range(self.e.get_n_splits()):
            if (node, i) in self.result:
                for inner_idx, df in self.result[(node, i)].items():
                    df = df['feature_importances']
                    df = df.to_frame() if type(df) == pd.Series else df.copy()
                    df.columns = pd.MultiIndex.from_product([[i], [inner_idx], df.columns])
                    dfs.append(df)
        return pd.concat(dfs, axis=1)

    def get_tree(self, node, idx=0, inner_idx=0):
        if (node, idx) not in self.result:
            self.set_node(node)

        return self.result[(node, idx)][inner_idx]['tree']
