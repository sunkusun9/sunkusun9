"""
LightGBM adapter
"""

import pandas as pd
from ._base import ModelAdapter

def create_progress_callback(n_estimators, period_pct, logger):
    last_printed = [-1]  # mutable object
    def callback(env):
        current = env.iteration + 1
        percentage = (current / n_estimators) * 100

        # period_pct마다 출력
        if int(percentage / (period_pct * 100)) > last_printed[0]:
            last_printed[0] = int(percentage / (period_pct * 100))

            # metric 정보 추출
            metrics_str = ""
            if env.evaluation_result_list:
                last_metrics = []
                for item in env.evaluation_result_list:
                    dataset_name, metric_name, value, _ = item
                    last_metrics.append(f"{dataset_name}-{metric_name}: {value:.4f}")
                metrics_str = ", ".join(last_metrics)
            logger.adhoc_progress(current, n_estimators, metrics_str if metrics_str else None)

    return callback

class LightGBMAdapter(ModelAdapter):
    """Adapter for LightGBM models (LGBMClassifier, LGBMRegressor)

    LightGBM도 eval_set 파라미터를 사용하지만 약간 다른 방식입니다.
    """

    def get_fit_params(self, X_train, y_train=None, X_eval=None, y_eval=None, params=None, logger = None):
        """LightGBM의 fit 파라미터 구성"""
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
                # n_estimators 추출 (기본값 100)
                n_estimators = params.get('n_estimators', 100) if params else 100
                callbacks = fit_params.get('callbacks', [])
                if logger is not None:
                    callbacks.append(create_progress_callback(n_estimators, self.verbose, logger))
                fit_params['callbacks'] = callbacks
            else:
                # verbose >= 1: LightGBM 기본 verbose (iteration 단위)
                fit_params['verbose'] = int(self.verbose)

        return fit_params

    @staticmethod
    def _get_feature_importances(processor, importance_type):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.n_features_in_))

        return pd.Series(
            obj.booster_.feature_importance(importance_type=importance_type),
            index=input_vars, name = 'importance'
        )

    @staticmethod
    def _get_evals_result(processor):
        obj = processor.obj
        evals_result = obj.evals_result_ if hasattr(obj, 'evals_result_') else {}
        return pd.concat(
            [pd.DataFrame(v).stack().rename(k) for k, v in evals_result.items()], axis=1
        ).stack()

    @staticmethod
    def _get_trees(processor):
        obj = processor.obj
        dump = obj.booster_.dump_model()
        return dump.get('tree_info', [])

LightGBMAdapter.result_objs = {
    'feature_importances_pvc': (LightGBMAdapter._get_feature_importances, True),
    'evals_result': (LightGBMAdapter._get_evals_result, True),
    'trees': (LightGBMAdapter._get_trees, False)
}