import re
import pandas as pd
import numpy as np

class LinearRegressionAnalyzer:
    coef = 'coef'
    intercept = 'intercept'

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
            output_vars = list(processor.output_vars) if hasattr(processor, 'output_vars') and processor.output_vars is not None else []
            obj = processor.obj
            coef = obj.coef_
            if coef.ndim == 1:
                coef = coef.reshape(1, -1)

            coef_df = pd.DataFrame(
                coef,
                index=output_vars,
                columns=input_vars
            )

            intercept = obj.intercept_
            if np.isscalar(intercept):
                intercept = [intercept]

            intercept_s = pd.Series(
                intercept,
                index=output_vars,
                name='intercept'
            )

            inner_results[inner_idx] = {
                'coef': coef_df,
                'intercept': intercept_s
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

    def get_coef(self, node):
        if (node, 0) not in self.result:
            self.set_node(node)
        dfs = list()        
        for i in range(self.e.get_n_splits()):
            if (node, i) in self.result:
                for inner_idx, df in self.result[(node, i)]:
                    df = df['coef'].copy()
                    df.columns = pd.MultiIndex.from_product([[i], [inner_idx], df.columns])
                    dfs.append(df)
        return pd.concat(dfs, axis=1)

    def get_intercept(self, node, idx=None, inner_idx=0):
        if (node, 0) not in self.result:
            self.set_node(node)
        dfs = list()        
        for i in range(self.e.get_n_splits()):
            if (node, i) in self.result:
                for inner_idx, df in self.result[(node, i)]:
                    df = df['intercept']
                    df = df.to_frame() if type(df) == pd.Series else df.copy()
                    df.columns = pd.MultiIndex.from_product([[i], [inner_idx], df.columns])
                    dfs.append(df)
        return pd.concat(dfs, axis=1)

class LogisticRegressionAnalyzer:
    coef = 'coef'
    intercept = 'intercept'

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
            obj = processor.obj
            classes = list(obj.classes_) if hasattr(obj, 'classes_') else []

            coef = obj.coef_
            if coef.ndim == 1:
                coef = coef.reshape(1, -1)

            coef_df = pd.DataFrame(
                coef,
                index=np.arange(coef.shape[0]),
                columns=input_vars
            )

            intercept = obj.intercept_
            if np.isscalar(intercept):
                intercept = [intercept]

            intercept_s = pd.Series(
                intercept,
                index=np.arange(coef.shape[0]),
                name='intercept'
            )

            inner_results[inner_idx] = {
                'coef': coef_df,
                'intercept': intercept_s
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

    def get_coef(self, node):
        if (node, 0) not in self.result:
            self.set_node(node)
        dfs = list()        
        for i in range(self.e.get_n_splits()):
            if (node, i) in self.result:
                for inner_idx, df in self.result[(node, i)].items():
                    df = df['coef'].copy()
                    df.columns = pd.MultiIndex.from_product([[i], [inner_idx], df.columns])
                    dfs.append(df)
        return pd.concat(dfs, axis=1)

    def get_intercept(self, node):
        if (node, 0) not in self.result:
            self.set_node(node)
        dfs = list()        
        for i in range(self.e.get_n_splits()):
            if (node, i) in self.result:
                for inner_idx, df in self.result[(node, i)].items():
                    df = df['intercept']
                    df = df.to_frame() if type(df) == pd.Series else df.copy()
                    df.columns = pd.MultiIndex.from_product([[i], [inner_idx], df.columns])
                    dfs.append(df)
        return pd.concat(dfs, axis=1)
