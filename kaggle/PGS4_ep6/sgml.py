from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

import joblib
import dill
import pickle as pkl
import numpy as np
import pandas as pd
import gc
import os

try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

def get_X_from_transformer(transformers):
    X = list()
    for i in transformers:
        X.extend(i[-1])
    X = list(set(X))
    return X

def get_ohe_transformer(hparams):
    if 'X_ohe' in hparams:
        return ('ohe', OneHotEncoder(**hparams.get('ohe', {})), hparams['X_ohe'])
    return None

def get_mm_transformer(hparams):
    if 'X_mm' in hparams:
        return ('mm', MinMaxScaler(), hparams['X_mm'])
    return None

def get_std_transformer(hparams):
    if 'X_std' in hparams:
        return ('std', StandardScaler(), hparams['X_std'])
    return None

def get_tgt_transformer(hparams):
    if 'X_tgt' in hparams:
        return ('tgt', TargetEncoder(), hparams['X_tgt'])
    return None

def get_lda_transformer(hparams):
    lda = hparams.get('lda', {})
    if len(lda) == 0:
        return None
    X_lda, _, lda_transformers = get_transformers(lda)
    if len(lda_transformers) > 0:
        return (
            'lda', make_pipeline(
                ColumnTransformer(lda_transformers) if len(lda_transformers) > 1 else lda_transformers[1], 
                LinearDiscriminantAnalysis(**lda.get('hparams', {}))
            ), X_lda
        )
    return None

def get_tsvd_transformer(hparams):
    tsvd = hparams.get('tsvd', {})
    if len(tsvd) == 0:
        return None
    X_tsvd, _, tsvd_transformers = get_transformers(tsvd)
    if len(tsvd_transformers) > 0:
        return (
            'tsvd', make_pipeline(
                ColumnTransformer(tsvd_transformers) if len(tsvd_transformers) > 1 else tsvd_transformers[1], 
                TruncatedSVD(**tsvd.get('hparams', {}))
            ), X_tsvd
        )
    return None

def get_pca_transformer(hparams):
    pca = hparams.get('pca', {})
    if len(pca) == 0:
        return None
    X_pca, _, pca_transformers = get_transformers(pca)
    if len(pca_transformers) > 0:
        return (
            'pca', make_pipeline(
                ColumnTransformer(pca_transformers) if len(pca_transformers) > 1 else pca_transformers[1], 
                PCA(**pca.get('hparams', {}))
            ), X_pca
        )
    return None

def get_transformers(hparams):
    transformers = list()
    for proc in [
        get_mm_transformer, get_std_transformer, get_pca_transformer,
        get_ohe_transformer, get_tgt_transformer, get_lda_transformer,
        get_tsvd_transformer
    ]:
        transformer = proc(hparams)
        if transformer is not None:
            transformers.append(transformer)
    X_num = hparams.get('X_num', [])
    transformers.append(('pt', 'passthrough', X_num))
    X = get_X_from_transformer(transformers)
    return X, [], transformers

def get_cat_transformers_ord(hparams):
    X, _, transformers = get_transformers(hparams)
    X_cat = hparams.get('X_cat', [])
    if len(X_cat) > 0:
        transformers = [('cat', OrdinalEncoder(dtype='int'), X_cat)] + transformers
        X_cat_feature = np.arange(0, len(X_cat)).tolist()
    else:
        X_cat_feature = []
    return get_X_from_transformer(transformers), X_cat_feature, transformers

def get_cat_transformers_pt(hparams):
    X, _, transformers = get_transformers(hparams)
    X_cat = hparams.get('X_cat', [])
    if len(X_cat) > 0:
        transformers = [('cat', 'passthrough', X_cat)] + transformers
        X_cat_feature = ['cat__{}'.format(i) for i in X_cat]
    return get_X_from_transformer(transformers), X_cat_feature, transformers

def get_cat_transformers_ohe(hparams):
    X, _, transformers = get_transformers(hparams)
    X_cat = hparams.get('X_cat', [])
    if len(X_cat) > 0:
        transformers = [('cat', OneHotEncoder(**hparams.get('ohe', {})), X_cat)] + transformers
    return get_X_from_transformer(transformers), [], transformers

def is_empty_transformer(transformers):
    return transformers is None or len(transformers) == 0 or (len(transformers) == 1 and transformers[0][1] == 'passthrough')

def gb_valid_config(train_set, valid_set):
    return {}, {'eval_set': [train_set, valid_set] if valid_set is not None else [train_set]}

def gb_valid_config2(train_set, valid_set):
    return {}, {'eval_set': [valid_set] if valid_set is not None else [train_set]}

def pass_learning_result(m, train_result, preprocessor=None):
    if preprocessor is None:
        return m, train_result
    else:
        return make_pipeline(preprocessor, m), train_result

def m_learning_result(train_result):
    return train_result

def lgb_learning_result(train_result):
    """
    Process LightGBM model results to extract evaluation metrics and feature importances.

    This function extracts and formats the training evaluation metrics and feature importances 
    from a trained LightGBM model.

    Parameters:
        m (lightgbm.Booster or lightgbm.sklearn.LGBMModel): 
            The trained LightGBM model object that contains the evaluation results and feature importances.
        train_result (dict): 
            A dictionary containing training-related information, including the names of the features used.
        preprocessor (object, optional): 
            A preprocessing object if used in the model pipeline (not utilized in this function, but included for consistency).

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame or None: A DataFrame containing evaluation metrics for each dataset and metric, indexed by iteration. 
              If no evaluation results are present, returns None.
            - pd.Series: A Series of feature importances sorted in ascending order, indexed by feature names.
            - dict: The original `train_result` dictionary.
    """
    return {
        'valid_result': pd.concat([
            pd.DataFrame(
                train_result['model'].evals_result_[i]
            ).rename(columns=lambda x: (i, x)) for i in train_result['model'].evals_result_.keys()
        ], axis=1).pipe(
            lambda x: x.reindex(columns = pd.MultiIndex.from_tuples(x.columns.tolist(), names=['set', 'metric'])).swaplevel(axis=1)
        ) if len(train_result['model'].evals_result_) > 0 else None, 
        'feature_importance': pd.Series(train_result['model'].feature_importances_, index=train_result['variables']).sort_values(),
        **{k: v for k, v in train_result.items() if k != 'model'}
    }

def xgb_learning_result(train_result):
    return {
        'valid_result': pd.concat([
            pd.DataFrame(
                train_result['model'].evals_result_[i]
            ).rename(columns=lambda x: (i, x)) for i in train_result['model'].evals_result_.keys()
        ], axis=1).pipe(
            lambda x: x.reindex(columns = pd.MultiIndex.from_tuples(x.columns.tolist(), names=['set', 'metric'])).swaplevel(axis=1)
        ), 
        'feature_importance': pd.Series(
            train_result['model'].feature_importances_, index=train_result['variables']
        ).sort_values(),
        **{k: v for k, v in train_result.items() if k != 'model'}
    }

def cb_learning_result(train_result):
    return {
        'valid_result': pd.concat([
            pd.DataFrame(
                train_result['model'].evals_result_[i]
            ).rename(columns=lambda x: (i, x)) for i in train_result['model'].evals_result_.keys()
        ], axis=1).pipe(
            lambda x: x.reindex(columns = pd.MultiIndex.from_tuples(x.columns.tolist(), names=['set', 'metric'])).swaplevel(axis=1)
        ), 
        'feature_importance': pd.Series(
            train_result['model'].feature_importances_, index=train_result['variables']
        ).sort_values(),
        **{k: v for k, v in train_result.items() if k != 'model'}
    }

class LGBMFitProgressbar:
    def __init__(self, precision = 5, start_position=0, metric=None, greater_is_better = True):
        self.start_position = start_position
        self.fmt = '{:.' + str(precision) + 'f}'
        self.metric = metric
        self.metric_hist = list()
        self.greater_is_better = greater_is_better

    def __repr__(self):
        return 'LGBMFitProgressbar'
    
    def _init(self, env):
        self.total_iteration = env.end_iteration - env.begin_iteration
        self.progress_bar = tqdm(total=self.total_iteration, desc='Round', position=self.start_position, leave=False)

    def __call__(self, env):
        if env.iteration == env.begin_iteration:
            self._init(env)
        self.progress_bar.update(1)
        if env.evaluation_result_list is not None:
            results = list()
            for item in env.evaluation_result_list:
                if len(item) >= 3:
                    data_name, eval_name, result = item[:3]
                    results.append(
                        '{}_{}:{}'.format(data_name, eval_name, self.fmt.format(result))
                    )
                    if self.metric == '{}_{}'.format(data_name, eval_name):
                        self.metric_hist.append(result)
            if self.metric is not None:
                if self.greater_is_better:
                    results.append(
                        'Best {}: {}/{}'.format(self.metric, np.argmax(self.metric_hist) + 1, self.fmt.format(np.max(self.metric_hist)))
                    )
                else:
                    results.append(
                        'Best {}: {}/{}'.format(self.metric, np.argmin(self.metric_hist) + 1, self.fmt.format(np.min(self.metric_hist)))
                    )
            self.progress_bar.set_postfix_str(', '.join(results))
        if self.total_iteration - 1 == env.iteration - env.begin_iteration:
            self.progress_bar.close()
            del self.progress_bar
            self.progress_bar = None

try:
    import xgboost as xgb
    class XGBFitProgressbar(xgb.callback.TrainingCallback):
        def __init__(self, n_estimators, precision=5, start_position=0, metric=None, greater_is_better=True):
            self.start_position = start_position
            self.n_estimators = n_estimators
            self.fmt = '{:.' + str(precision) + 'f}'
            self.metric = metric
            self.metric_hist = []
            self.greater_is_better = greater_is_better
            self.progress_bar = None
        
        def __repr__(self):
            return 'XGBFitProgressbar'
        
        def before_training(self, model):
            self.progress_bar = tqdm(total=self.n_estimators, desc='Round', position=self.start_position, leave=False)
            return model
    
        def after_iteration(self, model, epoch, evals_log):
            # 진행 상태를 업데이트
            self.progress_bar.update(1)
    
            results = []
            for data_name, metrics in evals_log.items():
                for eval_name, eval_results in metrics.items():
                    result = eval_results[-1]
                    results.append(f'{data_name}_{eval_name}:{self.fmt.format(result)}')
                    if self.metric == f'{data_name}_{eval_name}':
                        self.metric_hist.append(result)
    
            if self.metric is not None and self.metric_hist:
                if self.greater_is_better:
                    best_round = np.argmax(self.metric_hist) + 1
                    best_value = np.max(self.metric_hist)
                else:
                    best_round = np.argmin(self.metric_hist) + 1
                    best_value = np.min(self.metric_hist)
    
                results.append(f'Best {self.metric}: {best_round}/{self.fmt.format(best_value)}')
    
            self.progress_bar.set_postfix_str(', '.join(results))
    
            # False를 반환하면 학습이 계속 진행됨
            return False
    
        def after_training(self, model):
            # 학습이 종료되면 진행바를 닫음
            self.progress_bar.close()
            del self.progress_bar
            self.progress_bar = None
            return model
except:
    pass

class CatBoostFitProgressbar:
    def __init__(self, n_estimators, precision=5, start_position=0, metric=None, greater_is_better=True):
        self.start_position = start_position
        self.n_estimators = n_estimators
        self.fmt = '{:.' + str(precision) + 'f}'
        self.metric = metric
        self.metric_hist = list()
        self.greater_is_better = greater_is_better
        self.progress_bar = None
    
    def __repr__(self):
            return 'CatBoostFitProgressbar'
    
    def after_iteration(self, info):
        if self.progress_bar is None:
            self.progress_bar = tqdm(total=self.n_estimators, desc='Round', position=self.start_position, leave=False)

        self.progress_bar.update(1)
        results = list()
        if info.metrics is not None:
            for k, v in info.metrics.items():
                results_2 = list()
                for k2, v2 in v.items():
                    results_2.append('{}: {}'.format(k2, self.fmt.format(v2[-1])))
                    if self.metric == f'{k}_{k2}':
                        self.metric_hist.append(v2[-1])
                results.append('{}: {}'.format(k, ', '.join(results_2)))

        if self.metric is not None and self.metric_hist and len(self.metric_hist) > 0:
            if self.greater_is_better:
                best_round = np.argmax(self.metric_hist) + 1
                best_value = np.max(self.metric_hist)
            else:
                best_round = np.argmin(self.metric_hist) + 1
                best_value = np.min(self.metric_hist)

            results.append(f'Best {self.metric}: {best_round}/{self.fmt.format(best_value)}')
        
        self.progress_bar.set_postfix_str(', '.join(results))
        if self.progress_bar.n == self.n_estimators:
            self.after_train()
        return True

    def after_train(self):
        if self.progress_bar is not None:
            self.progress_bar.close()
            del self.progress_bar
            self.progress_bar = None


def train_model(model, model_params, df_train, X, y, valid_splitter=None, preprocessor=None, fit_params={}, valid_config_proc = None, target_func=None, **argv):
    """
    Train a model
    Parameters:
        model: Class
            Model class
        model_param: dict
            Model hyper parameters
        df_train: pd.DataFrame
            Train data
        X: list
            input variable names
        y: str
            target variable
        valid_splitter: function
            splitter to generate evaluation set for early stopping
        preprocessor: sklearn.preprocessing. 
            preprocessor. it will be connected using make_pipeline
        fit_params: dict
            parameters for fit
        valid_config_proc: function
            validation configuration function
    Returns
        dict
        train_resu;t
    Examples
    >>> X_cont =['Diameter', 'Whole weight.2', 'Whole weight.1', 'Shell weight', 'Length', 'Height_n', 'Whole weight']
    >>> X_cat = ['Sex']
    >>> X_all = X_cont + X_cat    
    >>> def gb_valid_config(train_set, valid_set):
    >>>    return {}, {'eval_set': [train_set, valid_set] if valid_set is not None else [train_set]}
    
    >>> train_model(lgb.LGBMRegressor, {'verbose': -1}, df_train_sp, X_all, 'target',
    >>>     valid_splitter = lambda x: train_test_split(x, train_size=0.9, stratify=x['Rings'], random_state=123),
    >>>     fit_params={'categorical_feature': ['Sex'], 'callbacks': [lgb.early_stopping(5, verbose=False)]},
    >>>     valid_config_proc=sgml.gb_valid_config
    >>> )
    """
    df_valid, X_valid = None, None
    result = {}
    if preprocessor is not None:
        preprocessor = clone(preprocessor)
        if valid_splitter is not None:
            df_train, df_valid = valid_splitter(df_train)
        X_train = preprocessor.fit_transform(df_train[X], df_train[y])
        result['variables'] = preprocessor.get_feature_names_out()
        if valid_splitter is not None:
            X_valid = preprocessor.transform(df_valid[X])
    else:
        if valid_splitter is not None:
            df_train, df_valid = valid_splitter(df_train)
            X_valid = df_valid[X]
        X_train = df_train[X]
        result['variables'] = X.copy()
    if target_func is None:
        y_train = df_train[y]
        if df_valid is not None:
            y_valid = df_valid[y]
    else:
        y_train = target_func(df_train, df_train[y])
        if df_valid is not None:
            y_valid = target_func(df_valid, df_valid[y])
    if valid_config_proc is not None:
        if X_valid is not None:
            model_params_2, fit_params_2 = valid_config_proc((X_train, y_train), (X_valid, y_valid))
            result['valid_shape'] = X_valid.shape
        else:
            model_params_2, fit_params_2 = valid_config_proc((X_train, y_train), None)
    else:
        model_params_2, fit_params_2 = {}, {}
    result['train_shape'] = X_train.shape
    result['target'] = y
    result['target_func'] = target_func
    m =  model(**model_params, **model_params_2)
    m.fit(X_train, y_train, **fit_params, **fit_params_2)
    del X_train, y_train
    if df_valid is not None:
        del X_valid, y_valid, df_valid
    gc.collect()
    result['model'] = m
    if preprocessor is not None:
        result['preprocessor'] = preprocessor
    return result

class BaseCallBack:
    def start(self, n_splits):
        pass
    def start_fold(self, fold):
        pass
    def end_fold(self, fold, train_metrics, valid_metrics, model_result_cv):
        pass
    def end(self):
        pass

class ProgressCallBack(BaseCallBack):
    def __init__(self, precision=5, start_position=0):
        self.start_position = start_position
        self.fmt = '{:.' + str(precision) + 'f}'
        self.progress_bar = None
    
    def start(self, n_splits):
        self.progress_bar = tqdm(total=n_splits, desc='Fold', position=self.start_position, leave=False)
    
    def end_fold(self, fold, train_metrics, valid_metrics, model_result_cv):
        self.progress_bar.update(1)
        results = list()
        if len(train_metrics) > 0:
            results.append(
                '{}±{}'.format(self.fmt.format(np.mean(train_metrics)), self.fmt.format(np.std(train_metrics)))
            )
        results.append(
            '{}±{}'.format(self.fmt.format(np.mean(valid_metrics)), self.fmt.format(np.std(valid_metrics)))
        )
        self.progress_bar.set_postfix_str(', '.join(results))
    def end(self):
        self.progress_bar.close()
        del self.progress_bar
        self.progress_bar = None

def cv_model(sp, model, model_params, df, X, y, predict_func, score_func, return_train_scores = True,
            preprocessor=None, result_proc=None, train_data_proc=None, train_params={}, sp_y=None, groups=None, 
            target_func=None, target_invfunc=None, progress_callback=None):
    """
    Train a model
    Parameters:
        sp: Splitter
            the instance of sklearn splitter
        model: Class
            Model class
        model_param: dict
            Model hyper parameters
        df: pd.DataFrame
            cross-validate data
        X: list
            input variable names
        y: str
            target variable
        predict_func: function
            prediction function
        score_func: function
            score function
        return_train_scores: bool
            return train scores
        preprocessor: sklearn.preprocessing. 
            preprocessor. it will be connected using make_pipeline
        fit_params: dict
            parameters for fit
        sp_y: str
            splitter y value name
        target_func: function
            function to transform the target
        target_inv_func: function
            inverse function to transform the predicted target
    Returns
        list, list, Series, list
        train_metrics, valid_metrics, s_prd, model_result_cv
    Example
    >>> X_cont =['Diameter', 'Whole weight.2', 'Whole weight.1', 'Shell weight', 'Length', 'Height_n', 'Whole weight']
    >>> X_cat = ['Sex']
    >>> X_all = X_cont + X_cat
    >>> def predict(m, df_valid, X):
    >>>     return pd.Series(m.predict(df_valid[X]), index=df_valid.index)
    >>> def score_func(y_true, prds):
    >>>     return -(mean_squared_error(y_true.sort_index(), prds.sort_index()) ** 0.5)
    >>> def gb_valid_config(train_set, valid_set):
    >>>     return {}, {'eval_set': [train_set, valid_set] if valid_set is not None else [train_set]}
    >>> cv_model(StratifiedKFold(n_splits=5, random_state=123, shuffle=True), 
    >>>         lgb.LGBMRegressor, {'verbose': -1}, df_train_sp, X_all, 'target',
    >>>         predict_func=predict, scores = score_func,
    >>>         train_params={
    >>>             'valid_splitter': lambda x: train_test_split(x, train_size=0.9, stratify=x['Rings'], random_state=123),
    >>>             'fit_params': {'categorical_feature': ['Sex'], 'callbacks': [lgb.early_stopping(5, verbose=False)]},
    >>>             'valid_config_proc': sgml.gb_valid_config
    >>>         }, sp_y = 'Rings'
    >>> )
    """
    train_scores, valid_scores = list(), list()
    valid_prds = list()
    if sp_y is None:
        sp_y = y
    model_result = list()
    sp_params = {'X': df[X], 'y': df[sp_y], 'groups': None if groups is None else df[groups]}
    if progress_callback is None:
        progress_callback = BaseCallBack()
    progress_callback.start(sp.get_n_splits(**sp_params))
    for fold, (train_idx, valid_idx) in enumerate(sp.split(**sp_params)):
        progress_callback.start_fold(fold)
        df_cv_train, df_valid = df.iloc[train_idx], df.iloc[valid_idx]
        if train_data_proc != None:
            df_cv_train = train_data_proc(df_cv_train)
        result = train_model(model, model_params, df_cv_train, X, y, preprocessor=preprocessor, target_func=target_func, **train_params)
        if 'preprocessor' in result:
            m = make_pipeline(result['preprocessor'], result['model'])
        else:
            m = result['model']
        if target_invfunc is None:
            target_invfunc = lambda x: x
        valid_prds.append(target_invfunc(predict_func(m, df_valid, X)))
        if return_train_scores:
            train_scores.append(
                score_func(df_cv_train, target_invfunc(predict_func(m, df_cv_train, X)))
            )
        del m
        valid_scores.append(score_func(df_valid, valid_prds[-1]))
        if result_proc is not None:
            model_result.append(result_proc(result))
        progress_callback.end_fold(fold, train_scores, valid_scores, model_result)
    s_prd = pd.concat(valid_prds, axis=0)
    progress_callback.end()
    ret = {'valid_scores': valid_scores, 'valid_prd': s_prd, 'model_result': model_result}
    if return_train_scores:
        ret['train_scores'] = train_scores
    return ret

def cv(df, sp, hparams, config, adapter, **argv):
    if 'validation_splitter' in config:
        argv['validation_splitter'] = config.pop('validation_splitter')
    return cv_model(
        sp=sp, df=df, **config, **adapter.adapt(hparams, is_train=False, **argv)
    )

def train(df, hparam, config, adapter, **argv):
    hparam_ = adapter.adapt(hparam, is_train=True, **argv)
    train_params = hparam_.pop('train_params') if 'train_params' in hparam_ else {}
    return train_model(df_train=df, **hparam_, **config, **train_params), hparam_['X']

def stack_cv(cv_list, y):
    return pd.concat([
        i.cv_best_['prd'].rename(i.name) for i in cv_list
    ] + [y], axis=1, join='inner')

def stack_prd(cv_list, df, config):
    return pd.concat([
        i.get_predictor()(df).rename(i.name) for i in cv_list
    ], axis=1)

class BaseAdapter():
    def save_model(self, filename, model):
        joblib.dump(model, filename)
        
    def load_model(self, filename):
        return joblib.load(filename)

class SklearnAdapter(BaseAdapter):
    def __init__(self, model):
        self.model = model

    def adapt(self, hparams, is_train=False, **argv):
        X, _, transformers = get_transformers(hparams)
        return {
            'model': self.model,
            'model_params': hparams.get('model_params', {}),
            'X': X,
            'preprocessor': ColumnTransformer(transformers) if not is_empty_transformer(transformers) else None,
            'result_proc': argv.get('result_proc', None)
        }

class LGBMAdapter(BaseAdapter):
    def __init__(self, model):
        self.model = model

    def adapt(self, hparams, is_train=False, **argv):
        X, X_cat_feature, transformers = get_cat_transformers_ord(hparams)
        validation_fraction = hparams.get('validation_fraction', 0)
        if validation_fraction > 0:
            if argv.get('validation_splitter', None) is None:
                validation_splitter = lambda x: train_test_split(x, test_size=validation_fraction, random_state=123)
            else:
                validation_splitter = argv.get('validation_splitter')(validation_fraction)
        else:
            validation_splitter = None
        return {
            'model': self.model, 
            'model_params': {'verbose': -1, **hparams['model_params']},
            'X': X,
            'preprocessor': ColumnTransformer(transformers) if not is_empty_transformer(transformers) else None,
            'train_params': {
                'fit_params': {
                    'categorical_feature': X_cat_feature,
                    'callbacks': [LGBMFitProgressbar()]
                },
                'valid_splitter': validation_splitter,
                'valid_config_proc': gb_valid_config,
            },
            'result_proc': argv.get('result_proc', lgb_learning_result),
        }

class XGBAdapter(BaseAdapter):
    def __init__(self, model, target_func=None):
        self.model = model
        self.target_func = target_func

    def adapt(self, hparams, is_train=False, **argv):
        X, _, transformers = get_cat_transformers_ohe(hparams)
        validation_fraction = hparams.get('validation_fraction', 0)
        if validation_fraction > 0:
            if argv.get('validation_splitter', None) is None:
                validation_splitter = lambda x: train_test_split(x, test_size=validation_fraction, random_state=123)
            else:
                validation_splitter = argv.get('validation_splitter')(validation_fraction)
        else:
            validation_splitter = None
        return {
            'model': self.model, 
            'model_params': {
                **hparams['model_params'], 
                'callbacks': [XGBFitProgressbar(n_estimators=hparams['model_params'].get('n_estimators', 100))]},
            'X': X,
            'preprocessor': ColumnTransformer(transformers) if not is_empty_transformer(transformers) else None,
            'train_params': {
                'valid_splitter': validation_splitter,
                'valid_config_proc': gb_valid_config,
                'fit_params': {'verbose': False}
            },
            'result_proc': argv.get('result_proc', xgb_learning_result),
            'target_func': argv.get('target_func', self.target_func)
        }

class CBAdapter(BaseAdapter):
    def __init__(self, model):
        self.model = model

    def adapt(self, hparams, is_train=False, **argv):
        X, X_cat_feature, transformers = get_cat_transformers_pt(hparams)
        validation_fraction = hparams.get('validation_fraction', 0)
        if validation_fraction > 0:
            if argv.get('validation_splitter', None) is None:
                validation_splitter = lambda x: train_test_split(x, test_size=validation_fraction, random_state=123)
            else:
                validation_splitter = argv.get('validation_splitter')(validation_fraction)
        else:
            validation_splitter = None
        return {
            'model': self.model, 
            'model_params': {
                **hparams['model_params'], 
                'cat_features': X_cat_feature, 'verbose': False},
            'X': X,
            'preprocessor': ColumnTransformer(transformers).set_output(transform='pandas') if not is_empty_transformer(transformers) else None,
            'train_params': {
                'valid_splitter': validation_splitter,
                'valid_config_proc': gb_valid_config,
                'fit_params': {'callbacks': [CatBoostFitProgressbar(n_estimators=hparams['model_params'].get('n_estimators', 100))]}
            },
            'result_proc': argv.get('result_proc', cb_learning_result)
        }

class CVModel:
    def __init__(self, path, name, sp, config, adapter):
        self.path = path
        self.name = name
        self.sp = sp
        self.adapter = adapter
        self.config = config
        self.cv_results_ = dict()
        self.cv_best_= {
            'score': -np.inf, 'hparams': {}, 'prd': None, 'k': ''
        }
        self.train_ = {
            'hparams': {}, 'k': '', 'result': {}, 'X': ''
        }
        self.preprocessor_ = None
        self.model_ = None

    def adhoc(self, df, sp, hparam):
        return cv(df, sp, hparam, self.config, self.adapter)

    def cv(self, df, hparams, rerun=False, **argv):
        k = str(hparams)
        if k in self.cv_results_ and not rerun:
            return self.cv_results_[k]
        result = cv(df, self.sp, hparams, self.config, self.adapter, **argv)
        score =  np.mean(result['valid_scores'])
        prd = result.pop('valid_prd')
        self.cv_results_[k] = result
        if score > self.cv_best_['score']:
            self.cv_best_['score'] = score
            self.cv_best_['hparams'] = hparams.copy()
            self.cv_best_['prd'] = prd
            self.cv_best_['k'] = k
        self.save()
        return result

    def get_best_result(self):
        return self.cv_results_[
            self.cv_best_['k']
        ]
    
    def train(self, df, rerun=False, **argv):
        if self.train_['k'] == self.cv_best_['k'] and not rerun:
            return self.train_['result']
        result, X = train(df, self.cv_best_['hparams'], self.config, self.adapter, **argv)
        if 'preprocessor' in result:
            self.preprocessor_ = result.pop('preprocessor')
            joblib.dump(self.preprocessor_, os.path.join(self.path, self.name + '.pre'))
        else:
            self.preprocessor_ = None
        self.model_ = result.pop('model')
        self.adapter.save_model(os.path.join(self.path, self.name + '.model'), self.model_)
        self.train_['hparams'] = self.cv_best_['hparams']
        self.train_['k'] = self.cv_best_['k']
        self.train_['result'] = result
        self.train_['X'] = X
        self.save()
        return result

    def get_predictor(self):
        if self.train_['k'] == '' or self.train_['k'] != self.cv_best_['k']:
            return None

        if self.model_ == None:
            self.preprocessor_, self.model_ = CVModel.load_predictor(self.path, self.name, self.adapter)
            
        if self.preprocessor_ is None:
            model = self.model_
        else:
            model = make_pipeline(self.preprocessor_, self.model_)

        return lambda x: self.config['predict_func'](model, x, self.train_['X'])

    def load_predictor(path, name, adapter):
        if os.path.exists(os.path.join(path,  name + '.pre')):
            preprocessor_ = joblib.load(os.path.join(path,  name + '.pre'))
        else:
            preprocessor_ = None
        model_ = adapter.load_model(os.path.join(path,  name + '.model'))
        return preprocessor_, model_
    
    def load(path, name):
        with open(os.path.join(path,  name + '.cv'), 'rb') as f:
            obj = dill.load(f)
        cv_obj = CVModel(path, name, obj['sp'], obj['config'], obj['adapter'])
        cv_obj.cv_results_ = obj['cv_results_']
        cv_obj.cv_best_ = obj['cv_best_']
        cv_obj.train_ = obj['train_']
        return cv_obj

    def save(self):
        with open(os.path.join(self.path,  self.name + '.cv'), 'wb') as f:
            dill.dump({
                'adapter': self.adapter,
                'sp': self.sp,
                'config': self.config,
                'cv_results_': self.cv_results_,
                'cv_best_': self.cv_best_,
                'train_': self.train_
            }, f)