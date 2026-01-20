import re
import pandas as pd
import numpy as np

class LDAAnalyzer:
    coef = 'coef'
    intercept = 'intercept'
    scalings = 'scalings'
    explained_variance_ratio = 'explained_variance_ratio'

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
            classes = list(processor.classes_) if hasattr(processor, 'classes_') else []
            output_vars = list(processor.output_vars) if hasattr(processor, 'output_vars') and processor.output_vars is not None else []

            coef = processor.coef_
            if coef.ndim == 1:
                coef = coef.reshape(1, -1)

            coef_df = pd.DataFrame(
                coef,
                index=classes,
                columns=input_vars
            )

            intercept = processor.intercept_
            if np.isscalar(intercept):
                intercept = [intercept]

            intercept_s = pd.Series(
                intercept,
                index=classes,
                name='intercept'
            )

            scalings_df = pd.DataFrame(
                processor.scalings_,
                index=input_vars,
                columns=output_vars if output_vars else [f'LD{i}' for i in range(processor.scalings_.shape[1])]
            )

            explained_variance_ratio_s = pd.Series(
                processor.explained_variance_ratio_,
                index=output_vars if output_vars else [f'LD{i}' for i in range(len(processor.explained_variance_ratio_))],
                name='explained_variance_ratio'
            )

            inner_results[inner_idx] = {
                'coef': coef_df,
                'intercept': intercept_s,
                'scalings': scalings_df,
                'explained_variance_ratio': explained_variance_ratio_s
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

    def get_coef(self, node, idx=None, inner_idx=0):
        if (node, 0) not in self.result:
            self.set_node(node)

        if idx is None:
            dfs = []
            for i in range(self.e.get_n_splits()):
                if (node, i) in self.result:
                    df = self.result[(node, i)][inner_idx]['coef'].copy()
                    df.columns = pd.MultiIndex.from_product([[i], df.columns])
                    dfs.append(df)
            return pd.concat(dfs, axis=1)
        return self.result[(node, idx)][inner_idx]['coef']

    def get_intercept(self, node, idx=None, inner_idx=0):
        if (node, 0) not in self.result:
            self.set_node(node)

        if idx is None:
            return pd.DataFrame({
                i: self.result[(node, i)][inner_idx]['intercept']
                for i in range(self.e.get_n_splits())
                if (node, i) in self.result
            })
        return self.result[(node, idx)][inner_idx]['intercept']

    def get_scalings(self, node, idx=None, inner_idx=0):
        if (node, 0) not in self.result:
            self.set_node(node)

        if idx is None:
            dfs = []
            for i in range(self.e.get_n_splits()):
                if (node, i) in self.result:
                    df = self.result[(node, i)][inner_idx]['scalings'].copy()
                    df.columns = pd.MultiIndex.from_product([[i], df.columns])
                    dfs.append(df)
            return pd.concat(dfs, axis=1)
        return self.result[(node, idx)][inner_idx]['scalings']

    def get_explained_variance_ratio(self, node, idx=None, inner_idx=0):
        if (node, 0) not in self.result:
            self.set_node(node)

        if idx is None:
            return pd.DataFrame({
                i: self.result[(node, i)][inner_idx]['explained_variance_ratio']
                for i in range(self.e.get_n_splits())
                if (node, i) in self.result
            })
        return self.result[(node, idx)][inner_idx]['explained_variance_ratio']
