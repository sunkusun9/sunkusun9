from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
import dproc

import lightgbm as lgb

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

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
                i: c[i] + X[i].value_counts().pipe(lambda x: x.loc[(x >= self.min_frequency) & ~x.index.isin(c[i])]).index.tolist()
                for i in X.columns
            }
        else:
            c = {
                i: c[i] + X[i].loc[~X[i].isin(c[i])].unique().tolist() for i in X.columns
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
        if len(self.transformers) == 0:
            return
        df_vars = pd.concat([
            pd.Series(t if type(t) == list else t.get_feature_names_out()).str.split('__', expand = True).rename(columns = lambda x: x + 1).pipe(
                lambda x: pd.concat([pd.Series(n, index = x.index, name = 'name'), x], axis=1)
            )
            for n, t in self.transformers
        ], axis = 0).reset_index(drop=True)
        
        self.df_vars_ = df_vars.join(
            pd.concat([
                pd.Series(t if type(t) == list else t.get_feature_names_out(), name='org')
                for _, t in self.transformers
            ]).reset_index(drop=True)
        ).set_index(df_vars.columns[:-1].tolist()).sort_index()

    def get_vars_list(self):
        return self.df_vars_

    def get_vars(self, name = None):
        if name is None:
            return self.df_vars_['org'].tolist()
        return self.df_vars_.loc[name, 'org'].tolist()

    def fit(self, X, y = None):
        for _, i in self.transformers:
            if type(i) == list:
                continue
            i.fit(X, y)
        if self.target is not None and self.target in X.columns:
            self.label_transformer.fit(X[self.target])
        self._get_features()
    
    def transform(self, X):
        lbl = [] if self.target is None or self.target not in X.columns else [
            pd.Series(self.label_transformer.transform(X[self.target]), index=X.index, name=self.target) if self.label_transformer is not None else X[self.target]
        ]
        return pd.concat([
            (X[i] if type(i) == list else i.transform(X))
            for name, i in self.transformers
        ] + lbl, axis=1)

    def append(self, name, transformer, X = None):
        for i, t in enumerate(self.transformers):
            if t[0] == name:
                del self.transformers[i]
                break
        self.transformers.append((name, transformer))
        self._get_features()
        if X is None:
            return None
        
        if type(transformer) == list:
            return X[transformer]
        else:
            return transformer.transform(X)

    def clear(self):
        self.transformers = list()
        self._get_featues()
    
    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return self.get_vars()

class AttachTransformer(TransformerMixin):
    def __init__(self, body_transformer, transformers):
        self.body_transformer = body_transformer
        self.transformers = list(transformers)
        self._get_features()
        
    def _get_features(self):
        if len(self.transformers) == 0:
            return
        df_vars = pd.concat([
            pd.Series(t if type(t) == list else t.get_feature_names_out()).str.split('__', expand = True).rename(columns = lambda x: x + 1).pipe(
                lambda x: pd.concat([pd.Series(n, index = x.index, name = 'name'), x], axis=1)
            )
            for n, t in self.body_transformer.transformers + self.transformers
        ], axis = 0).reset_index(drop=True)

        self.df_vars_ = df_vars.join(
            pd.concat([
                pd.Series(t if type(t) == list else t.get_feature_names_out(), name='org')
                for _, t in self.body_transformer.transformers + self.transformers
            ]).reset_index(drop=True)
        ).set_index(df_vars.columns[:-1].tolist()).sort_index()

    def get_vars_list(self):
        return self.df_vars_

    def get_vars(self, name = None):
        if name is None:
            return self.df_vars_['org'].tolist()
        return self.df_vars_.loc[name, 'org'].tolist()

    def fit(self, X, y = None):
        X = self.body_transform.fit_transform(X, y)
        for _, i in self.transformers:
            if type(i) == list:
                continue
            i.fit(X, y)
        self._get_features()
    
    def transform(self, X):
        X_ = self.body_transformer.transform(X)
        return pd.concat(
            [X_] + [
            (X_[i] if type(i) == list else i.transform(X_))
            for name, i in self.transformers
        ], axis=1)

    def append(self, name, transformer, X = None):
        for i, t in enumerate(self.transformers):
            if t[0] == name:
                del self.transformers[i]
                break
        self.transformers.append((name, transformer))
        self._get_features()
        if X is None:
            return None
        if type(transformer) == list:
            return X[transformer]
        else:
            return transformer.transform(X)
            
    def clear(self):
        self.transformers = list()
        self._get_featues()

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return self.get_vars()

    def refresh(self):
        self._get_features()

class LGBMImputer(TransformerMixin):
    def __init__(self, lgb_model, hparams, X_num, X_cat, target, na_value = np.nan):
        self.lgb_model = lgb_model
        self.hparams = hparams
        self.X_num = X_num
        self.X_cat = X_cat
        self.target = target
        self.na_value = na_value

    def fit(self, X, y = None):
        self.model_ = self.lgb_model(verbose = -1, **self.hparams)
        X.loc[X[self.target] != self.na_value].pipe(
            lambda x: self.model_.fit(x[self.X_num + self.X_cat], x[self.target], categorical_feature = self.X_cat)
        )
        return self
    def transform(self, X):
        s = X[self.target].copy()
        s.loc[s == self.na_value] = X.loc[s == self.na_value].pipe(
            lambda x: pd.Series(self.model_.predict(x[self.X_num + self.X_cat]), index = x.index)
        )
        return s.to_frame()
    def get_params(self, deep=True):
        return {
            'lgb_model': self.lgb_model, 
            'hparams': self.hparams,
            'X_num': self.X_num,
            'X_cat': self.X_cat,
            'target': self.target,
            'na_value': self.na_value
        }

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return [target]

class ImputerProgressBar():
    def __init__(self):
        self.iter_progress_bar = None
        self.step_progress_bar = None
        self.start_position = 0
        self.total_iter = 0

    def on_train_begin(self):
        pass
        
    def on_iter_begin(self, total_iter, logs=None):
        if self.iter_progress_bar is None:
            self.iter_progress_bar = tqdm(total=total_iter, desc='Iteration', position=self.start_position, leave=False)
        if self.step_progress_bar is not None:
            self.step_progress_bar.reset()

    def on_step_begin(self, total_step, logs=None):
        if self.step_progress_bar is None:
            self.step_progress_bar = tqdm(total=total_step, desc="Step", position=self.start_position + 1, leave=False)
        
    def on_step_end(self, total_step, logs=None):
        self.step_progress_bar.update(1)

    def on_iter_end(self, i, logs=None):
        if self.iter_progress_bar is not None:
            self.iter_progress_bar.update(1)

    def on_train_end(self):
        if self.step_progress_bar is not None:
            self.step_progress_bar.close()
            del self.step_progress_bar
            self.step_progress_bar = None
        if self.iter_progress_bar is not None:
            self.iter_progress_bar.close()
            del self.iter_progress_bar
            self.iter_progress_bar = None

class LGBMIterativeImputer(TransformerMixin):
    def __init__(self, hparams, X_num, X_cat, targets, na_value = np.nan, hparams_dic = {}, na_value_dic = {}, 
                 n_iter = 2, validation_fraction = 0, progress_callback=ImputerProgressBar()):
        self.hparams = hparams
        self.X_num = X_num
        self.X_cat = X_cat
        self.targets = targets
        self.na_value = na_value
        self.hparams_dic = hparams_dic
        self.na_value_dic = na_value_dic
        self.n_iter = n_iter
        self.validation_fraction = validation_fraction
        self.models_ = None
        self.hist_ = None
        self.progress_callback = progress_callback

    def fit(self, X, y = None):
        self.models_ = list()
        self.hist_ = list()
        self.progress_callback.on_train_begin()
        for i in range(self.n_iter):
            self.progress_callback.on_iter_begin(self.n_iter)
            self.partial_fit(X, y)
            self.progress_callback.on_iter_end(i)
        self.progress_callback.on_train_end()
        return self

    def partial_fit(self, X, y = None):
        if self.models_ is None:
            self.models_ = list()
            self.hist_ = list()
        if len(self.models_) >= self.n_iter:
            return self
        round_1 = list()
        round_hist = list()
        X_ = X.copy()
        self.progress_callback.on_step_begin(len(self.targets))
        for target in self.targets:
            val = [i for i in X.columns if i != target]
            hparams = self.hparams_dic.get(target, self.hparams)
            na_value = self.na_value_dic.get(target, self.na_value)
            if target in self.X_cat:
                round_1.append(
                    lgb.LGBMClassifier(verbose = - 1, **hparams)
                )
                if self.validation_fraction <= 0:
                    X_train = X_.loc[X_[target] != na_value]
                    X_test = None
                else:
                    X_train, X_test = X_.loc[X_[target] != na_value].pipe(
                        lambda x: train_test_split(x, test_size = self.validation_fraction, random_state = 123, stratify = x[target])
                    )
            else:
                round_1.append(
                    lgb.LGBMRegressor(verbose = -1, **hparams)
                )
                if self.validation_fraction <= 0:
                    X_train = X_.loc[X_[target] != na_value]
                    X_test = None
                else:
                    X_train, X_test = X_.loc[X_[target] != na_value].pipe(
                        lambda x: train_test_split(x, test_size = self.validation_fraction, random_state = 123)
                    )
            round_1[-1].fit(X_train[val], X_train[target])
            X_.loc[X_[target] == na_value, target] = round_1[-1].predict(X_.loc[X[target] == na_value, val])
            if X_test is not None:
                if target in self.X_cat:
                    round_hist.append(
                        accuracy_score(X_test[target], round_1[-1].predict(X_test[val]))
                    )
                else:
                    round_hist.append(
                        mean_squared_score(X_test[target], round_1[-1].predict(X_test[val]))
                    )
            self.progress_callback.on_step_end(len(self.targets))
        self.models_.append(round_1)
        self.hist_.append(round_hist)
        del X_
        return self
    
    def transform(self, X, n_iter = None):
        if n_iter is None:
            n_iter = len(self.models_)
        X_ = X.copy()
        for i, round_1 in enumerate(self.models_):
            if i >= n_iter:
                break
            for target, m in zip(self.targets, round_1):
                na_value = self.na_value_dic.get(target, self.na_value)
                val = [i for i in X.columns if i != target]
                X_.loc[X[target] == na_value, target] = m.predict(X_.loc[X[target] ==  na_value, val])
        return X_[self.targets]

    def get_params(self, deep=True):
        return {
            'hparams':  hparams,
            'X_num': X_num,
            'X_cat': X_cat,
            'targets': targets,
            'na_value': na_value,
            'hparams_dic': hparams_dic,
            'na_value_dic': na_value_dic,
            'n_iter': n_iter,
            'validate_fraction': validate_fraction
        }
    
    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return [target]


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
        return list(self.repl_dic_.keys())

class CatArrangerDics(TransformerMixin):
    def __init__(self, dics):
        self.dics = dics

    def fit(self, X, y = None):
        self.repl_dic_ = {
            i: {j: self.dics[i].get(j, j) for j in X[i].unique()}
            for i in X.columns if i in self.dics
        }
        return self
        
    def transform(self, X):
        return pd.DataFrame(
            pd.concat([
                dproc.replace_cat(X.loc[X[k].notna(), k], v).rename(k)
                for k, v in self.repl_dic_.items()
            ] + [X[i] for i in X.columns if i not in self.dics], axis=1), index = X.index
        )

    def get_params(self, deep=True):
        return {
            "dics": self.dics
        }

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return list(self.repl_dic_.keys())

class EvalTransformer(TransformerMixin):
    def __init__(self, expressions, local_dict = None):
        self.expressions = expressions
        self.local_dict = local_dict

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return pd.concat([
            X.eval(j, local_dict = self.local_dict).rename(i)
            for i, j in self.expressions
        ], axis=1)

    def get_params(self, deep=True):
        return {
            'expressions': self.expressions, 
            'local_dict': self.local_dict
        }

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return [i for i, _ in self.expressions]