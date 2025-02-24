from sklearn.base import TransformerMixin
import pandas as pd
import dproc

class CIBMTTransformer(TransformerMixin):
    def __init__(self, X_2, X_4, X_na, X_nom):
        self.X_2 = X_2
        self.X_4 = X_4
        self.X_na = X_na
        self.X_nom = X_nom
        self.d_tri = pd.CategoricalDtype([-1, 0, 1], ordered = True)
    
    def fit(self, X, y = None):
        self.vars_ = X.columns.tolist() + [i + '_na' for i in self.X_na]
        return self
    
    def transform(self, X, y = None):
        return dproc.join_and_assign(
            X,
            pd.concat(
                [
                    X[v].isna().rename(v + '_na') for v in self.X_na
                ] + [
                    dproc.replace_cat(X[v], lambda x: x.lower()).rename(v).map(d).astype(self.d_tri).fillna(0).astype('int8') for v, d in self.X_2
                ] + [
                    dproc.replace_cat(X[v], lambda x: x.lower()).rename(v).map({'no': -1, 'yes': 1, 'not done': 0}).astype(self.d_tri).fillna(0).astype('int8') for v in self.X_4
                ] + [
                    dproc.replace_cat(X[v], lambda x: x.lower()).rename(v) for v in self.X_nom
                ], axis=1
            )
        )
    
    def get_params(self, deep=True):
        return {
            'X_2': self.X_2,
            'X_4': self.X_4,
            'X_na': self.X_na,
            'X_nom': self.X_nom
        }

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return self.vars_


def filter_censor_data(x, t):
    return x.loc[((x['efs_time'] < t) & (x['efs'] == 1)) | (x['efs_time'] >= t)]