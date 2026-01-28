from ._base import BaseAnalyzer
import pandas as pd
import numpy as np

class LMAnalyzer(BaseAnalyzer):
    
    @classmethod
    def coef(self, processor):
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
        self.results[node].append(
            pd.DataFrame(coef_, index=idx, columns=coef_name)
        )
