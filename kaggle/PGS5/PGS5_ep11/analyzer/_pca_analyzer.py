import re
import pandas as pd

class PCAAnalyzer:
    explained_variance = 'explained_variance'
    explained_variance_ratio = 'explained_variance_ratio'
    components = 'components'

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

            explained_variance = pd.Series(
                obj.explained_variance_,
                index=output_vars,
                name='explained_variance'
            )

            explained_variance_ratio = pd.Series(
                obj.explained_variance_ratio_,
                index=output_vars,
                name='explained_variance_ratio'
            )

            components = pd.DataFrame(
                obj.components_,
                index=output_vars,
                columns=input_vars
            )

            inner_results[inner_idx] = {
                'explained_variance': explained_variance,
                'explained_variance_ratio': explained_variance_ratio,
                'components': components
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

    def get_explained_variance(self, node):
        if (node, 0) not in self.result:
            self.set_node(node)
        dfs = list()        
        for i in range(self.e.get_n_splits()):
            if (node, i) in self.result:
                for inner_idx, df in self.result[(node, i)].items():
                    df = df['explained_variance']
                    df = df.to_frame() if type(df) == pd.Series else df.copy()
                    df.columns = pd.MultiIndex.from_product([[i], [inner_idx], df.columns])
                    dfs.append(df)
        return pd.concat(dfs, axis=1)

    def get_explained_variance_ratio(self, node):
        if (node, 0) not in self.result:
            self.set_node(node)
        
        dfs = list()        
        for i in range(self.e.get_n_splits()):
            if (node, i) in self.result:
                for inner_idx, df in self.result[(node, i)].items():
                    df = df['explained_variance_ratio']
                    df = df.to_frame() if type(df) == pd.Series else df.copy()
                    df.columns = pd.MultiIndex.from_product([[i], [inner_idx], df.columns])
                    dfs.append(df)
        return pd.concat(dfs, axis=1)

    def get_components(self, node):
        if (node, 0) not in self.result:
            self.set_node(node)

        dfs = list()        
        for i in range(self.e.get_n_splits()):
            if (node, i) in self.result:
                for inner_idx, df in self.result[(node, i)].items():
                    df = df['components']
                    df = df.to_frame() if type(df) == pd.Series else df.copy()
                    df.columns = pd.MultiIndex.from_product([[i], [inner_idx], df.columns])
                    dfs.append(df)
        return pd.concat(dfs, axis=1)
