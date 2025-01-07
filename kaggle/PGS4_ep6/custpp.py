import pandas as pd
import dproc
from sklearn.base import TransformerMixin


class CatNomTransformer(TransformerMixin):
    def __init__(self, mapper):
        self.mapper = mapper

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return pd.DataFrame(
            pd.concat([
                dproc.replace_cat(X.loc[X[k].isin(v.keys()), k].astype('category'), v).rename(k)
                for k, v in self.mapper.items()
            ], axis=1), index = X.index
        )

    def get_params(self, deep=True):
        return {
            "mapper": self.mapper
        }

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return list(self.mapper)