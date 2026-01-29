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

    def get_fit_params(self, X_train, y_train=None, X_eval=None, y_eval=None, params=None, logger = None):
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

    @staticmethod
    def _get_feature_importances_pvc(processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.feature_count_))

        return pd.Series(
            obj.get_feature_importance(type='PredictionValuesChange'),
            index=input_vars, name = 'PredictionValuesChange'
        )

    @staticmethod
    def _get_feature_importances_interaction(processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.feature_count_))

        interaction = obj.get_feature_importance(type='Interaction')
        return pd.DataFrame(
            interaction, columns=['feat1', 'feat2', 'importance']
        ).assign(
            feat1=lambda x: x['feat1'].astype('int').apply(lambda y: input_vars[y]),
            feat2=lambda x: x['feat2'].astype('int').apply(lambda y: input_vars[y]),
        ).set_index(['feat1', 'feat2'])['importance']

    @staticmethod
    def _get_evals_result(processor):
        obj = processor.obj
        evals_result = obj.get_evals_result() if hasattr(obj, 'get_evals_result') else {}
        return pd.concat(
            [pd.DataFrame(v).stack().rename(k) for k, v in evals_result.items()], axis=1
        ).stack()

    @staticmethod
    def _get_trees(processor):
        obj = processor.obj
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            obj.save_model(f.name, format="json")
            trees = json.load(f).get('oblivious_trees', [])
        return trees

CatBoostAdapter.result_objs = {
    'feature_importances_pvc': (CatBoostAdapter._get_feature_importances_pvc, True),
    'feature_importances_interaction': (CatBoostAdapter._get_feature_importances_interaction, True),
    'evals_result': (CatBoostAdapter._get_evals_result, True), 
    'trees': (CatBoostAdapter._get_trees, False)
}