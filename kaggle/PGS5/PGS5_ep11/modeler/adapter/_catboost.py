"""
CatBoost adapter
"""

import tempfile
import json
import pandas as pd
from ._base import ModelAdapter


class CatBoostAdapter(ModelAdapter):
    """Adapter for CatBoost models (CatBoostClassifier, CatBoostRegressor)

    CatBoost도 eval_set을 지원합니다.
    """

    result_objs = ['feature_importances_pvc', 'feature_importances_interaction', 'evals_result', 'trees']

    def get_fit_params(self, X_train, y_train=None, X_eval=None, y_eval=None, params=None):
        """CatBoost의 fit 파라미터 구성"""
        fit_params = {}

        # eval_set 구성
        if self.eval_mode and self.eval_mode != 'none' and X_eval is not None and y_eval is not None:
            if self.eval_mode == 'valid':
                fit_params['eval_set'] = [(X_eval, y_eval)]
            elif self.eval_mode == 'both':
                fit_params['eval_set'] = [(X_train, y_train), (X_eval, y_eval)]

        # verbose 처리
        if self.verbose > 0:
            if self.verbose < 1:
                # 0 < verbose < 1: 진행률 기반 출력
                # CatBoost는 복잡한 callback 구조라서 간단히 기본 verbose 사용
                fit_params['verbose'] = False
            else:
                # verbose >= 1: CatBoost 기본 verbose (iteration 단위)
                fit_params['verbose'] = int(self.verbose)
        else:
            # verbose == 0: 출력 안함
            fit_params['verbose'] = False

        return fit_params

    def get_result(self, processor, name):
        if name == 'feature_importances_pvc':
            return self._get_feature_importances_pvc(processor)
        elif name == 'feature_importances_interaction':
            return self._get_feature_importances_interaction(processor)
        elif name == 'evals_result':
            return self._get_evals_result(processor)
        elif name == 'trees':
            return self._get_trees(processor)
        raise ValueError(f"{name} Unsupported result")

    def _get_feature_importances_pvc(self, processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.feature_count_))

        return pd.DataFrame(
            [obj.get_feature_importance(type='PredictionValuesChange')],
            columns=input_vars
        )

    def _get_feature_importances_interaction(self, processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.feature_count_))

        interaction = obj.get_feature_importance(type='Interaction')
        return pd.DataFrame(
            interaction, columns=['feat1', 'feat2', 'importance']
        ).assign(
            feat1=lambda x: x['feat1'].astype('int').apply(lambda y: input_vars[y]),
            feat2=lambda x: x['feat2'].astype('int').apply(lambda y: input_vars[y]),
        )

    def _get_evals_result(self, processor):
        obj = processor.obj
        return obj.get_evals_result() if hasattr(obj, 'get_evals_result') else {}

    def _get_trees(self, processor):
        obj = processor.obj
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            obj.save_model(f.name, format="json")
            trees = json.load(f).get('oblivious_trees', [])
        return trees
