from ._base import ModelAdapter

import pandas as pd
import numpy as np


class LMAdapter(ModelAdapter):
    result_objs = ['coef']

    def get_result(self, processor, name):
        if name == 'coef':
            return self._get_coef(processor)
        raise ValueError(f"{name} Unsupported result")

    def _get_coef(self, processor):
        coef_ = processor.obj.coef_
        if len(coef_.shape) == 1:
            if hasattr(processor.obj, 'intercept_'):
                coef_ = np.expand_dims(np.concatenate([coef_, [processor.obj.intercept_]]), axis=0)
                coef_name = list(processor.X_) + ['intercept']
            else:
                coef_ = np.expand_dims(processor.obj.coef_, axis=0)
                coef_name = processor.X_
            idx = [0]
        else:
            if hasattr(processor.obj, 'intercept_'):
                coef_ = np.concatenate([coef_, np.expand_dims(processor.obj.intercept_, axis=0)], axis=1)
                coef_name = list(processor.X_) + ['intercept']
            else:
                coef_ = processor.obj.coef_
                coef_name = processor.X_
            idx = np.arange(coef_.shape[0])
        return pd.DataFrame(coef_, index=idx, columns=coef_name)


class PCAAdapter(ModelAdapter):
    result_objs = ['explained_variance', 'explained_variance_ratio', 'components']

    def get_result(self, processor, name):
        if name == 'explained_variance':
            return self._get_explained_variance(processor)
        elif name == 'explained_variance_ratio':
            return self._get_explained_variance_ratio(processor)
        elif name == 'components':
            return self._get_components(processor)
        raise ValueError(f"{name} Unsupported result")

    def _get_explained_variance(self, processor):
        obj = processor.obj
        output_vars = list(processor.output_vars) if hasattr(processor, 'output_vars') and processor.output_vars is not None else list(range(len(obj.explained_variance_)))
        return pd.DataFrame(
            [obj.explained_variance_],
            columns=output_vars
        )

    def _get_explained_variance_ratio(self, processor):
        obj = processor.obj
        output_vars = list(processor.output_vars) if hasattr(processor, 'output_vars') and processor.output_vars is not None else list(range(len(obj.explained_variance_ratio_)))
        return pd.DataFrame(
            [obj.explained_variance_ratio_],
            columns=output_vars
        )

    def _get_components(self, processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.components_.shape[1]))
        output_vars = list(processor.output_vars) if hasattr(processor, 'output_vars') and processor.output_vars is not None else list(range(obj.components_.shape[0]))
        return pd.DataFrame(
            obj.components_,
            index=output_vars,
            columns=input_vars
        )


class LDAAdapter(ModelAdapter):
    result_objs = ['coef', 'intercept', 'scalings', 'explained_variance_ratio']

    def get_result(self, processor, name):
        if name == 'coef':
            return self._get_coef(processor)
        elif name == 'intercept':
            return self._get_intercept(processor)
        elif name == 'scalings':
            return self._get_scalings(processor)
        elif name == 'explained_variance_ratio':
            return self._get_explained_variance_ratio(processor)
        raise ValueError(f"{name} Unsupported result")

    def _get_coef(self, processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.coef_.shape[-1]))
        classes = list(processor.classes_) if hasattr(processor, 'classes_') and processor.classes_ is not None else list(range(obj.coef_.shape[0] if obj.coef_.ndim > 1 else 1))

        coef = obj.coef_
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)

        return pd.DataFrame(
            coef,
            index=classes,
            columns=input_vars
        )

    def _get_intercept(self, processor):
        obj = processor.obj
        classes = list(processor.classes_) if hasattr(processor, 'classes_') and processor.classes_ is not None else list(range(len(np.atleast_1d(obj.intercept_))))

        intercept = obj.intercept_
        if np.isscalar(intercept):
            intercept = [intercept]

        return pd.DataFrame(
            [intercept],
            columns=classes
        )

    def _get_scalings(self, processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.scalings_.shape[0]))
        output_vars = list(processor.output_vars) if hasattr(processor, 'output_vars') and processor.output_vars is not None else [f'LD{i}' for i in range(obj.scalings_.shape[1])]

        return pd.DataFrame(
            obj.scalings_,
            index=input_vars,
            columns=output_vars
        )

    def _get_explained_variance_ratio(self, processor):
        obj = processor.obj
        output_vars = list(processor.output_vars) if hasattr(processor, 'output_vars') and processor.output_vars is not None else [f'LD{i}' for i in range(len(obj.explained_variance_ratio_))]

        return pd.DataFrame(
            [obj.explained_variance_ratio_],
            columns=output_vars
        )


class DecisionTreeAdapter(ModelAdapter):
    result_objs = ['feature_importances', 'tree']

    def get_result(self, processor, name):
        if name == 'feature_importances':
            return self._get_feature_importances(processor)
        elif name == 'tree':
            return self._get_tree(processor)
        raise ValueError(f"{name} Unsupported result")

    def _get_feature_importances(self, processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(len(obj.feature_importances_)))

        return pd.DataFrame(
            [obj.feature_importances_],
            columns=input_vars
        )

    def _get_tree(self, processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.n_features_in_))

        tree_ = obj.tree_
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

        return pd.DataFrame(tree_structure)