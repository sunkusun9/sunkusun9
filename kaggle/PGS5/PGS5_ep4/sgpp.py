from sklearn.base import TransformerMixin
import pandas as pd
import polars as pl
import numpy as np
import dproc

import lightgbm as lgb

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

class ApplyWrapper(TransformerMixin):
    def __init__(self, transformer, vals, suffix = None, postfix = None):
        self.vals = vals
        self.transformer = transformer
        self.suffix = suffix
        self.postfix = postfix

    def fit(self, X, y = None):
        self.transformer.fit(X[self.vals], y)
        self.fitted_ = True
        return self

    def transform(self, X, **argv):
        if self.suffix is None and self.postfix is None:
            return dproc.join_and_assign(
                X, self.transformer.transform(X[self.vals])
            )
        if self.suffix is not None:
            return X.join(
                self.transformer.transform(X[self.vals], **argv).rename(columns = lambda x: self.suffix + x)
            )
        if self.postfix is not None:
            return X.join(
                self.transformer.transform(X[self.vals], **argv).rename(columns = lambda x: x + self.postfix)
            )
        return X
    
    def get_params(self, deep=True):
        return {
            "vals": self.vals, 
            "transformer": self.transformer,
            "suffix": self.suffix,
            "postfix": self.postfix
        }

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        vals = self.vals.copy()
        if self.suffix is not None:
            vals = [self.suffix + i for i in self.vals]
        if self.postfix is not None:
            vals = [i + self.postfix for i in self.vals]
        return vals
        
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
        self.fitted_ = True
        return self
        
    def transform(self, X):
        if self.unknown_value is not None or self.na_value is not None:
            if self.unknown_value is not None:
                ret = pd.concat([
                    dproc.rearrange_cat(X[k], v, lambda d, c: 0 if c not in d else c, use_set = True).rename(k)
                    for k, v in self.c_types_.items()
                ], axis = 1)
                if self.na_value is not None:
                    return ret.fillna(self.na_value)
            else:
                ret = pd.concat([
                    X[k].astype(v) for k, v in self.c_types_.items()
                ], axis = 1)
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

class CatArrangerFreqNearest(TransformerMixin):
    def __init__(self, min_frequency, na_value = None):
        self.min_frequency = min_frequency
        self.na_value = na_value

    def fit(self, X, y = None):
        if self.na_value is not None:
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
        self.fitted_ = True
        return self
        
    def transform(self, X):
        if self.na_value is not None:
            ret = pd.concat([
                dproc.rearrange_cat(X[k], v, lambda d, c: np.argmin(d - c) if c not in d else c, use_set = False).rename(k)
                for k, v in self.c_types_.items()
            ], axis = 1)
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

class CatOOVFilter(TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y = None):
        self.s_dtype_ = {i: X[i].dtype for i in X.columns}
        self.s_mode_ = X.apply(lambda x: x.mode()[0])
        self.fitted_ = True
        return self
    
    def transform(self, X):
        return pd.concat([
            dproc.rearrange_cat(X[k], v, lambda d, c: 0 if c not in d else c, use_set = True).rename(k)
            for k, v in self.s_dtype_.items()
        ], axis=1)
    def get_params(self, deep=True):
        return {
        }

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return list(self.s_dtype_.keys())
    
class FrequencyEncoder(TransformerMixin):
    def __init__(self, na_frequency = 0, dtype = 'int'):
        self.na_frequency = 0
        self.dtype = dtype

    def fit(self, X, y = None):
        self.freq_ = {
            i: X[i].value_counts()
            for i in X.columns
        }
        self.fitted_ = True
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

class CatCombiner(TransformerMixin):
    def __init__(self, combine_features):
        self.combine_features = combine_features

    def fit(self, X, y = None):
        self.fitted_ = True
        return self
        
    def transform(self, X):
        return pd.concat([
            dproc.combine_cat(X[i]).rename(j)
            for i, j in self.combine_features
        ], axis = 1)
    
    def get_params(self, deep=True):
        return {
            "combine_features": self.combine_features, 
        }

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return [i for _, i in self.combine_features]

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
        self.fitted_ = True
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
    def __init__(self, hparams, X_num, X_cat, targets, na_value = None, hparams_dic = {}, na_value_dic = {}, 
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
        self.initial_cat_values_ = None
        self.progress_callback = progress_callback

    def fit(self, X, y = None):
        self.models_ = list()
        self.hist_ = list()
        self.progress_callback.on_train_begin()
        X_ = X.copy()
        if len(self.X_cat) > 0:
            self.initial_cat_values_ = X_[self.X_cat].apply(lambda x: x.mode().iloc[0])
            X_ = dproc.join_and_assign(
                X_, X_[self.X_cat].fillna(self.initial_cat_values_)
            )
        for i in range(self.n_iter):
            self.progress_callback.on_iter_begin(self.n_iter)
            self._partial_fit(X_, X)
            self.progress_callback.on_iter_end(i)
        self.progress_callback.on_train_end()
        del X_
        self.fitted_ = True
        return self

    def _partial_fit(self, X, X_org):
        if self.models_ is None:
            self.models_ = list()
            self.hist_ = list()
        if len(self.models_) >= self.n_iter:
            return self
        round_1 = list()
        round_hist = list()
        self.progress_callback.on_step_begin(len(self.targets))
        for target in self.targets:
            val = [i for i in self.X_num + self.X_cat if i != target]
            hparams = self.hparams_dic.get(target, self.hparams)
            na_value = self.na_value_dic.get(target, self.na_value)
            s_tgt_notna = X_org[target].notna() if na_value is None else X_org[target] != na_value
            if target in self.X_cat:
                round_1.append(
                    lgb.LGBMClassifier(verbose = - 1, **hparams)
                )
                if self.validation_fraction <= 0:
                    X_train = X.loc[s_tgt_notna]
                    X_test = None
                else:
                    X_train, X_test = X.loc[s_tgt_notna].pipe(
                        lambda x: train_test_split(x, test_size = self.validation_fraction, random_state = 123, stratify = x[target])
                    )
            else:
                round_1.append(
                    lgb.LGBMRegressor(verbose = -1, **hparams)
                )
                if self.validation_fraction <= 0:
                    X_train = X.loc[s_tgt_notna]
                    X_test = None
                else:
                    X_train, X_test = X.loc[s_tgt_notna].pipe(
                        lambda x: train_test_split(x, test_size = self.validation_fraction, random_state = 123)
                    )
            round_1[-1].fit(X_train[val], X_train[target])
            X.loc[~s_tgt_notna, target] = X.loc[~s_tgt_notna, val].pipe(lambda x: pd.Series(round_1[-1].predict(x), index = x.index, dtype = X[target].dtype))
            if X_test is not None:
                if target in self.X_cat:
                    round_hist.append(
                        accuracy_score(X_test[target], round_1[-1].predict(X_test[val]))
                    )
                else:
                    round_hist.append(
                        mean_squared_error(X_test[target], round_1[-1].predict(X_test[val]))
                    )
            self.progress_callback.on_step_end(len(self.targets))
        self.models_.append(round_1)
        self.hist_.append(round_hist)
        return self
    
    def transform(self, X, n_iter = None):
        if n_iter is None:
            n_iter = len(self.models_)
        X_ = X.copy()
        if len(self.X_cat) > 0:
            self.initial_cat_values_ = X_[self.X_cat].apply(lambda x: x.mode().iloc[0])
            X_ = dproc.join_and_assign(
                X_, X_[self.X_cat].fillna(self.initial_cat_values_)
            )
        for i, round_1 in enumerate(self.models_):
            if i >= n_iter:
                break
            for target, m in zip(self.targets, round_1):
                na_value = self.na_value_dic.get(target, self.na_value)
                s_tgt_na = X[target].isna() if na_value is None else X[target] == na_value
                na_value = self.na_value_dic.get(target, self.na_value)
                val = [i for i in self.X_num + self.X_cat if i != target]
                X_.loc[s_tgt_na, target] = X_.loc[s_tgt_na, val].pipe(lambda x: pd.Series(m.predict(x), index = x.index, dtype = X[target].dtype))
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
        self.fitted_ = True
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
        self.fitted_ = True
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
        self.fitted_ = True
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

class TypeCaster(TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y = None):
        self.names_ = X.columns.tolist()
        self.fitted_ = True
        return self

    def transform(self, X):
        return X.astype(self.dtype)

    def get_params(self, deep=True):
        return {
            'dtype': self.dtype, 
        }

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return self.names_

class PolarsProcessor(TransformerMixin):
    def __init__(self, predefined_types = {}, read_method = 'read_csv'):
        self.predefined_types = predefined_types
        self.read_method = read_method

    def fit(self, X, y = None):
        if type(X) is list:
            self.df_type_ = dproc.merge_type_df([
                pl.scan_csv(i).pipe(dproc.get_type_df) for i in X
            ])
        else:
            self.df_type_ = pl.scan_csv(X).pipe(dproc.get_type_df)
        self.pl_type_ = dproc.get_type_pl(
            self.df_type_, self.predefined_types
        )
        self.fitted_ = True
        return self

    def transform(self, X):
        if type(X) is list:
            return pl.concat([
                getattr(pl, self.read_method)(i, schema_overrides = self.pl_type_) for i in X
            ])
        else:
            return getattr(pl, self.read_method)(X, schema_overrides = self.pl_type_)

    def get_params(self, deep = True):
        return {
            'predefined_types': self.predefined_types,
            'read_method': self.read_method
        }
        
    def set_output(self, transform = 'polars'):
        pass

    def get_feature_names_out(self, X = None):
        return self.pl_type
    
class ExprProcessor(TransformerMixin):
    def __init__(self, dict_expr, with_columns = True):
        self.dict_expr = dict_expr
        self.with_columns = with_columns
        self.columns = []

    def fit(self, X, y = None):
        if self.with_columns:
            self.columns = X.columns
        self.fitted_ = True
        return self
        
    def transform(self, X):
        if self.with_columns:
            return X.with_columns(**self.dict_expr)
        else:
            return X.select(**self.dict_expr)

    def get_params(self, deep=True):
        return {
            'dict_expr': self.dict_expr,
            'with_columns': self.with_columns
        }

    def set_output(self, transform = 'polars'):
        pass

    def get_feature_names_out(self, X = None):
        return self.columns + list(self.dict_expr.keys())

class PandasConverter(TransformerMixin):
    def __init__(self, index_col = None):
        self.index_col = index_col
        self.columns_ = None

    def fit(self, X, y = None):
        self.columns_ = [i for i in X.columns if i != self.index_col]
        self.fitted_ = True
        return self

    def transform(self, X):
        df = X.to_pandas()
        if self.index_col is not None and self.index_col in df.columns:
            df = df.set_index(self.index_col)
        return df

    def get_params(self, deep = True):
        return {
            'index_col': self.index_col
        }

    def set_output(self, transform = 'pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return self.columns_


class JoinEncoder(TransformerMixin):
    def __init__(self, data, on = None):
        self.data = data
        self.on = on
    def fit(self, X, y = None):
        self.features = X.columns.tolist()
        if type(self.data) == pd.Series:
            self.features.append(self.data.name)
        else:
            self.features.extend(self.data.columns.tolist())
        self.fitted_ = True
        return self
    
    def transform(self, X):
        return X.join(
            self.data, on = self.on
        )
    def get_params(self, deep = True):
        return {
            'data': self.data,
            'on': self.on
        }
        
    def set_output(self, transform = 'pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return self.features

class ColumnNameCleaner(TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y = None):
        self.columns_ = [i.strip().replace(' ', '_').replace('\t', '_') for i in X.columns]
        self.fitted_ = True
        return self

    def transform(self, X):
        if type(X)  == pd.DataFrame:
            return X.rename(columns = lambda x: x.strip().replace(' ', '_').replace('\t', '_'))
        else:
            return X.rename(lambda x: x.strip().replace(' ', '_').replace('\t', '_'))
    def get_params(self, deep = True):
        return {}
    def set_output(self, transdorm = 'pandas'):
        pass
    def get_feature_names_out(self, X = None):
        return self.columns_
                            
