"""
Keras adapter
"""

from ._base import ModelAdapter


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
