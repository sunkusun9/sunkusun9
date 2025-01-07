from sklearn.base import TransformerMixin
import pandas as pd
import dproc

class CatArrangerFreq(TransformerMixin):
    def __init__(self, min_frequency, unknown_value = None, na_value = None):
        self.min_frequency = min_frequency
        self.unknown_value = unknown_value
        self.na_value = na_value

    def fit(self, X, y = None):
        if self.unknown_value is not None or self.na_value is not None:
            if self.unknown_value == self.na_value:
                c = {
                    i: [self.unknown_value]
                    for i in X.columns
                }
            elif self.unknown_value is not None and self.na_value is not None:
                c = {
                    i: [self.unknown_value, self.na_value]
                    for i in X.columns
                }
            elif self.unknown_value is not None:
                c = {
                    i: [self.unknown_value]
                    for i in X.columns
                }
            else:
                c = {
                    i: [self.na_value]
                    for i in X.columns
                }
        else:
            c = {i: [] for i in X.columns}

        if self.min_frequency > 0:
            c = {
                i: X[i].value_counts().pipe(lambda x: x.loc[x >= self.min_frequency]).index.tolist() + c[i]
                for i in X.columns
            }
        else:
            c = {
                i: X[i].unique().tolist() + c[i] for i in X.columns
            }

        self.c_types_ = {
            i: pd.CategoricalDtype(c[i]) for i in X.columns
        }
        return self
        
    def transform(self, X):
        if self.unknown_value is not None or self.na_value is not None:
            if self.unknown_value is not None:
                ret = pd.concat([
                    dproc.rearrange_cat(X[k], v, lambda d, c: 0 if c not in d else c, use_set = True).rename(k)
                    for k, v in self.c_types_.items()
                ], axis = 1)
            else:
                ret = X
            if self.na_value is not None:
                return ret.fillna(self.na_value)
            return ret
        return X

    def get_params(self, deep=True):
        return {
            "min_frequency": self.min_frequency, 
            "unknown_value": self.unknown_value,
            "na_value": self.na_value
        }

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return list(self.c_types_)

class FrequencyEncoder(TransformerMixin):
    def __init__(self, na_frequency = 0, dtype = 'int'):
        self.na_frequency = 0
        self.dtype = dtype

    def fit(self, X, y = None):
        self.freq_ = {
            i: X[i].value_counts()
            for i in X.columns
        }
        return self
    def transform(self, X):
        return pd.concat([
            X[k].map(v).fillna(self.na_frequency).astype(self.dtype)
            for k, v in self.freq_.items()
        ], axis=1)

    def get_params(self, deep=True):
        return {
            'na_frequency': self.na_frequency, 
            'dtype': self.dtype
        }

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return list(self.freq_.keys())


class CombineTransformer(TransformerMixin):
    def __init__(self, transformers, target = None, label_transformer = None):
        self.transformers = list(transformers)
        self.target = target
        self.label_transformer = label_transformer
        self._get_features()
        
    def _get_features(self):
        df_vars = pd.concat([
            pd.Series(t.get_feature_names_out()).str.split('__', expand = True).rename(columns = lambda x: x + 1).pipe(
                lambda x: pd.concat([pd.Series(n, index = x.index, name = 'name'), x], axis=1)
            )
            for n, t in self.transformers
        ], axis = 0).reset_index(drop=True).assign(
            rename = ''
        )
        df_vars2 = df_vars.iloc[:, :-1].copy()
        while (df_vars2.shape[1] >= 1) & (df_vars2.shape[0] > 0):
            s_idx = df_vars2.groupby(df_vars2.iloc[:, -1]).transform('size') == 1
            if s_idx.sum() > 0:
                df_vars.loc[s_idx.loc[s_idx].index.values, 'rename'] = df_vars2.loc[s_idx].iloc[:, -1]
                df_vars2 = df_vars2.loc[~s_idx]
            a = df_vars2.iloc[:, -2:]
            df_vars2 = df_vars2.drop(columns = a.columns)
            if len(df_vars2) == 0:
                break
            df_vars2.insert(
                df_vars2.shape[1], 1, a.apply(lambda x: x.dropna().str.cat(sep='__'), axis=1)
            )
            
        self.df_vars_ = df_vars.join(
            pd.concat([
                pd.Series(t.get_feature_names_out(), name='org')
                for _, t in self.transformers
            ]).reset_index(drop=True)
        ).set_index(['name', 1, 2]).sort_index()

    def get_vars_list(self):
        return self.df_vars_

    def get_vars(self, name = None):
        if name is None:
            return self.df_vars_['rename'].tolist()
        return self.df_vars_.loc[name, 'rename'].tolist()

    def fit(self, X, y = None):
        for _, i in self.transformers:
            i.fit(X, y)
        if self.target is not None and self.target in X.columns:
            self.label_transformer.fit(X[self.target])
        self._get_features()
    
    def transform(self, X):
        lbl = [] if self.target is None or self.target not in X.columns else [
            pd.Series(self.label_transformer.transform(X[self.target]), index=X.index, name=self.target) if self.label_transformer is not None else X[self.target]
        ]
        return pd.concat([
            i.transform(X).rename(columns = self.df_vars_.loc[(name, ), ['org', 'rename']].set_index('org')['rename'])
            for name, i in self.transformers
        ] + lbl, axis=1)


class CatArrangerDic(TransformerMixin):
    def __init__(self, dic):
        self.dic = dic

    def fit(self, X, y = None):
        self.repl_dic_ = {
            i: {j: self.dic.get(j, j) for j in X[i].unique()}
            for i in X.columns
        }
        return self
        
    def transform(self, X):
        return pd.DataFrame(
            pd.concat([
                dproc.replace_cat(X.loc[X[k].notna(), k], v).rename(k)
                for k, v in self.repl_dic_.items()
            ], axis=1), index = X.index
        )

    def get_params(self, deep=True):
        return {
            "dic": self.dic
        }

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return list(self.c_types_)