"""
Model adapters for handling eval_set in different ML frameworks
"""

from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    """Abstract base class for model adapters

    각 머신러닝 프레임워크별로 eval_set 처리 방식이 다르므로,
    이를 통일된 인터페이스로 추상화합니다.
    """

    @abstractmethod
    def get_fit_params(self, X_train, y_train = None, X_eval=None, y_eval=None, params=None):
        """모델의 fit()에 전달할 파라미터를 구성

        Args:
            X_train: Training features (필수)
            y_train: Training target (Optional, default=None)
            X_eval: Evaluation features (Optional, default=None)
            y_eval: Evaluation target (Optional, default=None)
            params (dict): Processor에서 전달된 추가 파라미터 (Optional, default=None)
                          여기에 eval_mode도 포함되어 있음

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
        """XGBoost의 fit 파라미터 구성

        params에서 eval_mode를 추출하여 처리:
        - 'none' or None: eval_set 없이
        - 'valid': eval_set=[(X_eval, y_eval)]
        - 'both': eval_set=[(X_train, y_train), (X_eval, y_eval)]
        """
        fit_params = {}

        if params is None:
            return fit_params

        # eval_mode 추출
        eval_mode = params.get('eval_mode', None)

        # eval_set 구성
        if eval_mode and eval_mode != 'none' and X_eval is not None and y_eval is not None:
            if eval_mode == 'valid':
                fit_params['eval_set'] = [(X_eval, y_eval)]
            elif eval_mode == 'both':
                fit_params['eval_set'] = [(X_train, y_train), (X_eval, y_eval)]

        return fit_params


class LightGBMAdapter(ModelAdapter):
    """Adapter for LightGBM models (LGBMClassifier, LGBMRegressor)

    LightGBM도 eval_set 파라미터를 사용하지만 약간 다른 방식입니다.
    """

    def get_fit_params(self, X_train, y_train=None, X_eval=None, y_eval=None, params=None):
        """LightGBM의 fit 파라미터 구성"""
        fit_params = {}

        if params is None:
            return fit_params

        # eval_mode 추출
        eval_mode = params.get('eval_mode', None)

        # eval_set 구성
        if eval_mode and eval_mode != 'none' and X_eval is not None and y_eval is not None:
            if eval_mode == 'valid':
                fit_params['eval_set'] = [(X_eval, y_eval)]
            elif eval_mode == 'both':
                fit_params['eval_set'] = [(X_train, y_train), (X_eval, y_eval)]

        return fit_params


class CatBoostAdapter(ModelAdapter):
    """Adapter for CatBoost models (CatBoostClassifier, CatBoostRegressor)

    CatBoost도 eval_set을 지원합니다.
    """

    def get_fit_params(self, X_train, y_train=None, X_eval=None, y_eval=None, params=None):
        """CatBoost의 fit 파라미터 구성"""
        fit_params = {}

        if params is None:
            return fit_params

        # eval_mode 추출
        eval_mode = params.get('eval_mode', None)

        # eval_set 구성
        if eval_mode and eval_mode != 'none' and X_eval is not None and y_eval is not None:
            if eval_mode == 'valid':
                fit_params['eval_set'] = [(X_eval, y_eval)]
            elif eval_mode == 'both':
                fit_params['eval_set'] = [(X_train, y_train), (X_eval, y_eval)]

        return fit_params


class KerasAdapter(ModelAdapter):
    """Adapter for Keras models (KerasClassifier, KerasRegressor)

    Keras는 validation_data 파라미터로 (X, y) 튜플을 받습니다.
    """

    def get_fit_params(self, X_train, y_train=None, X_eval=None, y_eval=None, params=None):
        """Keras의 fit 파라미터 구성

        params에서 eval_mode를 추출하여 처리:
        - 'none' or None: validation_data 없이
        - 'valid' or 'both': validation_data=(X_eval, y_eval)
          (Keras는 train과 valid 동시 전달 불가)
        """
        fit_params = {}

        if params is None:
            return fit_params

        # eval_mode 추출
        eval_mode = params.get('eval_mode', None)

        # validation_data 구성
        if eval_mode and eval_mode != 'none' and X_eval is not None and y_eval is not None:
            # Keras는 'valid'와 'both' 모두 동일하게 처리 (validation_data만 지원)
            fit_params['validation_data'] = (X_eval, y_eval)

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


# Model adapter registry
# 모델 클래스명을 키로, 해당 어댑터 인스턴스를 값으로 매핑
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
    """모델 또는 모델명에 해당하는 어댑터를 반환

    Args:
        model_or_name: Model instance or model class name (str)

    Returns:
        ModelAdapter: Corresponding adapter instance, or DefaultAdapter if not found

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
