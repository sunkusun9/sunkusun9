"""
LightGBM adapter
"""

import pandas as pd
from ._base import ModelAdapter


class LightGBMAdapter(ModelAdapter):
    """Adapter for LightGBM models (LGBMClassifier, LGBMRegressor)

    LightGBM도 eval_set 파라미터를 사용하지만 약간 다른 방식입니다.
    """

    result_objs = ['feature_importances_split', 'feature_importances_gain', 'evals_result', 'trees']

    def get_fit_params(self, X_train, y_train=None, X_eval=None, y_eval=None, params=None):
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
                def create_progress_callback(n_estimators, period_pct):
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
                                metrics_str = " | " + ", ".join(last_metrics)

                            print(f"\r  Progress: {current}/{n_estimators} ({percentage:.1f}%){metrics_str}", end='', flush=True)

                    return callback

                # n_estimators 추출 (기본값 100)
                n_estimators = params.get('n_estimators', 100) if params else 100
                callbacks = fit_params.get('callbacks', [])
                callbacks.append(create_progress_callback(n_estimators, self.verbose))
                fit_params['callbacks'] = callbacks
            else:
                # verbose >= 1: LightGBM 기본 verbose (iteration 단위)
                fit_params['verbose'] = int(self.verbose)

        return fit_params

    def get_result(self, processor, name):
        if name == 'feature_importances_split':
            return self._get_feature_importances(processor, 'split')
        elif name == 'feature_importances_gain':
            return self._get_feature_importances(processor, 'gain')
        elif name == 'evals_result':
            return self._get_evals_result(processor)
        elif name == 'trees':
            return self._get_trees(processor)
        raise ValueError(f"{name} Unsupported result")

    def _get_feature_importances(self, processor, importance_type):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.n_features_in_))

        return pd.DataFrame(
            [obj.booster_.feature_importance(importance_type=importance_type)],
            columns=input_vars
        )

    def _get_evals_result(self, processor):
        obj = processor.obj
        evals_result = obj.evals_result_ if hasattr(obj, 'evals_result_') else {}
        return evals_result

    def _get_trees(self, processor):
        obj = processor.obj
        dump = obj.booster_.dump_model()
        return dump.get('tree_info', [])
