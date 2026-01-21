"""
XGBoost adapter
"""

import pandas as pd
from ._base import ModelAdapter


class XGBoostAdapter(ModelAdapter):
    """Adapter for XGBoost models (XGBClassifier, XGBRegressor)

    XGBoost는 eval_set 파라미터로 [(X, y), ...] 형태를 받습니다.
    """

    result_objs = [
        'feature_importances_weight', 'feature_importances_gain', 'feature_importances_cover',
        'feature_importances_total_gain', 'feature_importances_total_cover',
        'evals_result', 'trees'
    ]

    def get_params(self, params):
        """XGBoost 모델 생성자 파라미터 조정 (ProgressCallback 설정)"""
        if params is None:
            params = {}

        if self.verbose > 0 and self.verbose < 1:
            # 0 < verbose < 1: 진행률 기반 출력을 위한 callback 설정
            from xgboost.callback import TrainingCallback

            class ProgressCallback(TrainingCallback):
                def __init__(self, n_estimators, period_pct):
                    self.n_estimators = n_estimators
                    self.period_pct = period_pct
                    self.last_printed = -1

                def after_iteration(self, model, epoch, evals_log):
                    current = epoch + 1
                    percentage = (current / self.n_estimators) * 100

                    # period_pct마다 출력
                    if int(percentage / (self.period_pct * 100)) > self.last_printed:
                        self.last_printed = int(percentage / (self.period_pct * 100))

                        # metric 정보 추출
                        metrics_str = ""
                        if evals_log:
                            last_metrics = []
                            for dataset, metrics in evals_log.items():
                                for metric_name, values in metrics.items():
                                    last_metrics.append(f"{dataset}-{metric_name}: {values[-1]:.4f}")
                            metrics_str = " | " + ", ".join(last_metrics)

                        print(f"\r  Progress: {current}/{self.n_estimators} ({percentage:.1f}%){metrics_str}", end='', flush=True)

                    return False

            n_estimators = params.get('n_estimators', 100)
            callbacks = params.get('callbacks', [])
            callbacks.append(ProgressCallback(n_estimators, self.verbose))
            params['callbacks'] = callbacks

        return params

    def get_fit_params(self, X_train, y_train=None, X_eval=None, y_eval=None, params=None):
        """XGBoost의 fit 파라미터 구성"""
        fit_params = {}
        if params is not None and params.get('verbosity') > 0:
            fit_params['verbose'] = True
        else:
            fit_params['verbose'] = False
        # eval_set 구성
        if self.eval_mode and self.eval_mode != 'none' and X_eval is not None and y_eval is not None:
            if self.eval_mode == 'valid':
                fit_params['eval_set'] = [(X_eval, y_eval)]
            elif self.eval_mode == 'both':
                fit_params['eval_set'] = [(X_train, y_train), (X_eval, y_eval)]

        return fit_params

    def get_result(self, processor, name):
        if name == 'feature_importances_weight':
            return self._get_feature_importances(processor, 'weight')
        elif name == 'feature_importances_gain':
            return self._get_feature_importances(processor, 'gain')
        elif name == 'feature_importances_cover':
            return self._get_feature_importances(processor, 'cover')
        elif name == 'feature_importances_total_gain':
            return self._get_feature_importances(processor, 'total_gain')
        elif name == 'feature_importances_total_cover':
            return self._get_feature_importances(processor, 'total_cover')
        elif name == 'evals_result':
            return self._get_evals_result(processor)
        elif name == 'trees':
            return self._get_trees(processor)
        raise ValueError(f"{name} Unsupported result")

    def _get_feature_importances(self, processor, importance_type):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.n_features_in_))
        booster = obj.get_booster()
        scores = booster.get_score(importance_type=importance_type)

        return pd.DataFrame(
            [[scores.get(f, 0) for f in input_vars]],
            columns=input_vars
        )

    def _get_evals_result(self, processor):
        obj = processor.obj
        return obj.evals_result() if hasattr(obj, 'evals_result') else {}

    def _get_trees(self, processor):
        obj = processor.obj
        booster = obj.get_booster()
        return booster.trees_to_dataframe()
