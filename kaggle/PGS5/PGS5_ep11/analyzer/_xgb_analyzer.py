import re
import pandas as pd

class XGBAnalyzer:
    feature_importances_weight = 'feature_importances_weight'
    feature_importances_gain = 'feature_importances_gain'
    feature_importances_cover = 'feature_importances_cover'
    feature_importances_total_gain = 'feature_importances_total_gain'
    feature_importances_total_cover = 'feature_importances_total_cover'
    evals_result = 'evals_result'
    trees = 'trees'

    def __init__(self, e):
        self.e = e
        self.result = {}
        self.build_ids = {}

    def _get_importance(self, booster, importance_type, input_vars):
        scores = booster.get_score(importance_type=importance_type)
        return pd.Series(
            [scores.get(f, 0) for f in input_vars],
            index=input_vars,
            name=f'feature_importances_{importance_type}'
        )

    def _set_node(self, idx, node):
        build_id = self.build_ids.get((node, idx), '')
        current_build_id = ''.join([i['build_id'] for _, _, i in self.e.nodes[node].objs_[idx]])

        if build_id == current_build_id and (node, idx) in self.result:
            return

        self.build_ids[(node, idx)] = current_build_id

        inner_results = {}
        for inner_idx, (processor, _, _) in enumerate(self.e.nodes[node].objs_[idx]):
            input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else []
            booster = processor.get_booster()

            feature_importances_weight = self._get_importance(booster, 'weight', input_vars)
            feature_importances_gain = self._get_importance(booster, 'gain', input_vars)
            feature_importances_cover = self._get_importance(booster, 'cover', input_vars)
            feature_importances_total_gain = self._get_importance(booster, 'total_gain', input_vars)
            feature_importances_total_cover = self._get_importance(booster, 'total_cover', input_vars)

            evals_result = processor.evals_result() if hasattr(processor, 'evals_result') else {}

            trees = booster.trees_to_dataframe()

            inner_results[inner_idx] = {
                'feature_importances_weight': feature_importances_weight,
                'feature_importances_gain': feature_importances_gain,
                'feature_importances_cover': feature_importances_cover,
                'feature_importances_total_gain': feature_importances_total_gain,
                'feature_importances_total_cover': feature_importances_total_cover,
                'evals_result': evals_result,
                'trees': trees
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

    def get_feature_importances_weight(self, node):
        if (node, 0) not in self.result:
            self.set_node(node)

        dfs = list()
        for i in range(self.e.get_n_splits()):
            if (node, i) in self.result:
                for inner_idx, df in self.result[(node, i)].items():
                    df = df['feature_importances_weight']
                    df = df.to_frame() if type(df) == pd.Series else df.copy()
                    df.columns = pd.MultiIndex.from_product([[i], [inner_idx], df.columns])
                    dfs.append(df)
        return pd.concat(dfs, axis=1)

    def get_feature_importances_gain(self, node):
        if (node, 0) not in self.result:
            self.set_node(node)

        dfs = list()
        for i in range(self.e.get_n_splits()):
            if (node, i) in self.result:
                for inner_idx, df in self.result[(node, i)].items():
                    df = df['feature_importances_gain']
                    df = df.to_frame() if type(df) == pd.Series else df.copy()
                    df.columns = pd.MultiIndex.from_product([[i], [inner_idx], df.columns])
                    dfs.append(df)
        return pd.concat(dfs, axis=1)

    def get_feature_importances_cover(self, node):
        if (node, 0) not in self.result:
            self.set_node(node)

        dfs = list()
        for i in range(self.e.get_n_splits()):
            if (node, i) in self.result:
                for inner_idx, df in self.result[(node, i)].items():
                    df = df['feature_importances_cover']
                    df = df.to_frame() if type(df) == pd.Series else df.copy()
                    df.columns = pd.MultiIndex.from_product([[i], [inner_idx], df.columns])
                    dfs.append(df)
        return pd.concat(dfs, axis=1)

    def get_feature_importances_total_gain(self, node):
        if (node, 0) not in self.result:
            self.set_node(node)

        dfs = list()
        for i in range(self.e.get_n_splits()):
            if (node, i) in self.result:
                for inner_idx, df in self.result[(node, i)].items():
                    df = df['feature_importances_total_gain']
                    df = df.to_frame() if type(df) == pd.Series else df.copy()
                    df.columns = pd.MultiIndex.from_product([[i], [inner_idx], df.columns])
                    dfs.append(df)
        return pd.concat(dfs, axis=1)

    def get_feature_importances_total_cover(self, node):
        if (node, 0) not in self.result:
            self.set_node(node)

        dfs = list()
        for i in range(self.e.get_n_splits()):
            if (node, i) in self.result:
                for inner_idx, df in self.result[(node, i)].items():
                    df = df['feature_importances_total_cover']
                    df = df.to_frame() if type(df) == pd.Series else df.copy()
                    df.columns = pd.MultiIndex.from_product([[i], [inner_idx], df.columns])
                    dfs.append(df)
        return pd.concat(dfs, axis=1)

    def get_evals_result(self, node, idx=0, inner_idx=0):
        if (node, idx) not in self.result:
            self.set_node(node)

        return self.result[(node, idx)][inner_idx]['evals_result']

    def get_trees(self, node, idx=0, inner_idx=0):
        if (node, idx) not in self.result:
            self.set_node(node)

        return self.result[(node, idx)][inner_idx]['trees']
