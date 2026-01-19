"""
Model adapters for handling eval_set in different ML frameworks
"""

from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    """Abstract base class for model adapters

    각 머신러닝 프레임워크별로 eval_set 처리 방식이 다르므로,
    이를 통일된 인터페이스로 추상화합니다.
    """

    def __init__(self, eval_mode='both', verbose=0.1):
        """Adapter 초기화

        Args:
            eval_mode (str): Evaluation mode
                - 'none' or None: eval_set 없이
                - 'valid': validation set만 전달
                - 'both': train + validation set 전달
            verbose: Verbose 설정
                - 0: 출력 안함
                - 0 < verbose < 1: 전체 진행률을 % 단위로 표시 (주기: verbose * 100%)
                  예: 0.1이면 10%마다, 0.05면 5%마다
                - verbose >= 1: iteration 단위로 표시 (매 verbose번째 iteration)
                  예: 1이면 매 iteration마다, 10이면 매 10 iteration마다
        """
        self.eval_mode = eval_mode
        self.verbose = verbose

    @abstractmethod
    def get_fit_params(self, X_train, y_train = None, X_eval=None, y_eval=None, params=None):
        """모델의 fit()에 전달할 파라미터를 구성

        Args:
            X_train: Training features (필수)
            y_train: Training target (Optional, default=None)
            X_eval: Evaluation features (Optional, default=None)
            y_eval: Evaluation target (Optional, default=None)
            params (dict): Processor에서 전달된 추가 파라미터 (Optional, default=None)

        Returns:
            dict: fit()에 unpacking으로 전달할 파라미터
                  예: model.fit(**fit_params)
        """
        pass


class XGBoostAdapter(ModelAdapter):
    """Adapter for XGBoost models (XGBClassifier, XGBRegressor)

    XGBoost는 eval_set 파라미터로 [(X, y), ...] 형태를 받습니다.
    """

    def get_fit_params(self, X_train, y_train=None, X_eval=None, y_eval=None, params=None):
        """XGBoost의 fit 파라미터 구성"""
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

                            print(f"\r  Progress: {current}/{self.n_estimators} ({percentage:.1f}%){metrics_str}", flush=True)

                        return False

                # n_estimators 추출 (기본값 100)
                n_estimators = params.get('n_estimators', 100) if params else 100
                callbacks = fit_params.get('callbacks', [])
                callbacks.append(ProgressCallback(n_estimators, self.verbose))
                fit_params['callbacks'] = callbacks
            else:
                # verbose >= 1: XGBoost 기본 verbose (iteration 단위)
                fit_params['verbose'] = int(self.verbose)

        return fit_params


class LightGBMAdapter(ModelAdapter):
    """Adapter for LightGBM models (LGBMClassifier, LGBMRegressor)

    LightGBM도 eval_set 파라미터를 사용하지만 약간 다른 방식입니다.
    """

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


class CatBoostAdapter(ModelAdapter):
    """Adapter for CatBoost models (CatBoostClassifier, CatBoostRegressor)

    CatBoost도 eval_set을 지원합니다.
    """

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


class KerasAdapter(ModelAdapter):
    """Adapter for Keras models (KerasClassifier, KerasRegressor)

    Keras는 validation_data 파라미터로 (X, y) 튜플을 받습니다.
    """

    def get_fit_params(self, X_train, y_train=None, X_eval=None, y_eval=None, params=None):
        """Keras의 fit 파라미터 구성"""
        fit_params = {}

        # validation_data 구성
        if self.eval_mode and self.eval_mode != 'none' and X_eval is not None and y_eval is not None:
            # Keras는 'valid'와 'both' 모두 동일하게 처리 (validation_data만 지원)
            fit_params['validation_data'] = (X_eval, y_eval)

        # verbose 처리
        if self.verbose > 0:
            if self.verbose < 1:
                # 0 < verbose < 1: Keras의 verbose는 0, 1, 2만 지원하므로 1 사용
                fit_params['verbose'] = 1
            else:
                # verbose >= 1: Keras verbose (0: silent, 1: progress bar, 2: one line per epoch)
                fit_params['verbose'] = min(int(self.verbose), 2)
        else:
            # verbose == 0: 출력 안함
            fit_params['verbose'] = 0

        return fit_params


class DefaultAdapter(ModelAdapter):
    """Default adapter for models that don't support eval_set

    일반적인 sklearn 모델들을 위한 기본 어댑터
    eval_set을 지원하지 않으므로 일반 fit()만 수행
    """

    def get_fit_params(self, X_train, y_train=None, X_eval=None, y_eval=None, params=None):
        """기본 fit 파라미터 (추가 파라미터 없음)

        eval_set 관련 파라미터는 무시합니다.
        """
        # eval_set을 지원하지 않으므로 빈 dict 반환
        return {}


# Model adapter registry (인스턴스 저장)
# 모델 클래스명을 키로, 해당 어댑터 인스턴스를 값으로 매핑
# 기본 설정: eval_mode='both', verbose=0.1
MODEL_ADAPTERS = {
    'XGBClassifier': XGBoostAdapter(),
    'XGBRegressor': XGBoostAdapter(),
    'XGBRFClassifier': XGBoostAdapter(),
    'XGBRFRegressor': XGBoostAdapter(),

    'LGBMClassifier': LightGBMAdapter(),
    'LGBMRegressor': LightGBMAdapter(),
    'LGBMRanker': LightGBMAdapter(),

    'CatBoostClassifier': CatBoostAdapter(),
    'CatBoostRegressor': CatBoostAdapter(),
    'CatBoostRanker': CatBoostAdapter(),

    'KerasClassifier': KerasAdapter(),
    'KerasRegressor': KerasAdapter(),
}


def get_adapter(model_or_name):
    """모델 또는 모델명에 해당하는 어댑터 인스턴스를 반환

    Args:
        model_or_name: Model instance or model class name (str)

    Returns:
        ModelAdapter: Corresponding adapter instance, or DefaultAdapter instance if not found

    Example:
        >>> from xgboost import XGBClassifier
        >>> adapter = get_adapter(XGBClassifier)
        >>> # or
        >>> adapter = get_adapter('XGBClassifier')
    """
    if isinstance(model_or_name, str):
        model_name = model_or_name
    else:
        # model instance or class
        if hasattr(model_or_name, '__name__'):
            model_name = model_or_name.__name__
        else:
            model_name = model_or_name.__class__.__name__

    return MODEL_ADAPTERS.get(model_name, DefaultAdapter())


def register_adapter(model_name, adapter):
    """새로운 어댑터를 레지스트리에 등록

    Args:
        model_name (str): Model class name
        adapter (ModelAdapter): Adapter instance

    Example:
        >>> class MyCustomAdapter(ModelAdapter):
        ...     def get_fit_params(self, X_train, y_train, X_eval=None, y_eval=None, params=None):
        ...         # custom implementation
        ...         return {...}
        >>>
        >>> register_adapter('MyCustomModel', MyCustomAdapter())
    """
    if not isinstance(adapter, ModelAdapter):
        raise TypeError(f"adapter must be an instance of ModelAdapter, got {type(adapter)}")

    MODEL_ADAPTERS[model_name] = adapter
