from sklearn.base import clone

from sklearn.model_selection import train_test_split
from functools import partial

import joblib
import numpy as np
import pandas as pd
import gc
import os
import shap

try:
    from tqdm.notebook import tqdm
    from IPython.display import clear_output
except:
    from tqdm import tqdm
    clear_output = None


def pass_learning_result(train_result):
    return train_result


def predict_learning_result(train_result, df):
    return train_result['predictor'](df).values


def lgb_learning_result(train_result):
    """
    Process LightGBM model results to extract evaluation metrics and feature importances.

    This function extracts and formats the training evaluation metrics and feature importances 
    from a trained LightGBM model.

    Parameters:
        train_result (dict): 
            A dictionary containing training-related information, including the names of the features used.

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
            lambda x: x.reindex(columns=pd.MultiIndex.from_tuples(
                x.columns.tolist(), names=['set', 'metric'])).swaplevel(axis=1)
        ) if hasattr(train_result['model'], 'evals_result_') and len(train_result['model'].evals_result_) > 0 else None,
        'feature_importance': pd.Series(train_result['model'].feature_importances_, index=train_result['variables']).sort_values(),
        **{k: v for k, v in train_result.items() if k not in ['model', 'predictor']}
    }


def xgb_learning_result(train_result):
    return {
        'valid_result': pd.concat([
            pd.DataFrame(
                train_result['model'].evals_result_[i]
            ).rename(columns=lambda x: (i, x)) for i in train_result['model'].evals_result_.keys()
        ], axis=1).pipe(
            lambda x: x.reindex(columns=pd.MultiIndex.from_tuples(
                x.columns.tolist(), names=['set', 'metric'])).swaplevel(axis=1)
        ) if hasattr(train_result['model'], 'evals_result_') else None,
        'feature_importance': pd.Series(
            train_result['model'].feature_importances_, index=train_result['variables']
        ).sort_values(),
        **{k: v for k, v in train_result.items() if k not in ['model', 'predictor']}
    }


def cb_learning_result(train_result):
    return {
        'valid_result': pd.concat([
            pd.DataFrame(
                train_result['model'].evals_result_[i]
            ).rename(columns=lambda x: (i, x)) for i in train_result['model'].evals_result_.keys()
        ], axis=1).pipe(
            lambda x: x.reindex(columns=pd.MultiIndex.from_tuples(
                x.columns.tolist(), names=['set', 'metric'])).swaplevel(axis=1)
        ) if hasattr(train_result['model'], 'evals_result_') else None,
        'feature_importance': pd.Series(
            train_result['model'].feature_importances_, index=train_result['variables']
        ).sort_values(),
        **{k: v for k, v in train_result.items() if k not in ['model', 'predictor']}
    }


def gb_shap_learning_result(train_result, df, interaction=True):
    explainer = shap.TreeExplainer(train_result['model'])
    processor = train_result['preprocessor']
    result['shap_values'] = explainer.shap_values(df)
    if interaction:
        result['shap_interaction_values'] = explainer.shap_interaction_values(df)
    return result


def cb_interaction_importance(train_result):
    s_name = pd.Series(train_result['variables'])
    return pd.DataFrame(
        train_result['model'].get_feature_importance(type='Interaction'),
        columns=['Var1', 'Var2', 'Importance']
    ).assign(
        Var1=lambda x: x['Var1'].map(s_name),
        Var2=lambda x: x['Var2'].map(s_name),
    )


def lr_learning_result(train_result):
    return {
        'coef': pd.Series(train_result['model'].coef_, index=train_result['variables']) if len(train_result['model'].coef_.shape) == 1 else
        pd.DataFrame(train_result['model'].coef_.T,
                     index=train_result['variables'])
    } if type(train_result['model'].coef_) == np.ndarray else {
        'coef': pd.Series(train_result['model'].coef_.values, index=train_result['variables']) if len(train_result['model'].coef_.shape) == 1 else
        pd.DataFrame(train_result['model'].coef_.T.values,
                     index=train_result['variables'])
    }


class LGBMFitProgressbar:
    def __init__(self, precision=5, start_position=0, metric=None, greater_is_better=True, update_cycle=30):
        self.start_position = start_position
        self.fmt = '{:.' + str(precision) + 'f}'
        self.metric = metric
        self.metric_hist = list()
        self.greater_is_better = greater_is_better
        self.update_cycle = update_cycle

    def __repr__(self):
        return 'LGBMFitProgressbar'

    def _init(self, env):
        self.total_iteration = env.end_iteration - env.begin_iteration
        self.progress_bar = tqdm(
            total=self.total_iteration, desc='Round', position=self.start_position, leave=False)
        self.prog = 0

    def __call__(self, env):
        if env.iteration == env.begin_iteration:
            self._init(env)
        self.prog += 1
        if (self.prog % self.update_cycle) != 0:
            if self.total_iteration - 1 == env.iteration - env.begin_iteration:
                self.progress_bar.update(self.prog % self.update_cycle)
                self.progress_bar.close()
                del self.progress_bar
                self.progress_bar = None
            return
        self.progress_bar.update(self.update_cycle)
        if env.evaluation_result_list is not None:
            results = list()
            for item in env.evaluation_result_list:
                if len(item) >= 3:
                    data_name, eval_name, result = item[:3]
                    results.append(
                        '{}_{}:{}'.format(data_name, eval_name,
                                          self.fmt.format(result))
                    )
                    if self.metric == '{}_{}'.format(data_name, eval_name):
                        self.metric_hist.append(result)
            if self.metric is not None:
                if self.greater_is_better:
                    results.append(
                        'Best {}: {}/{}'.format(self.metric, np.argmax(
                            self.metric_hist) + 1, self.fmt.format(np.max(self.metric_hist)))
                    )
                else:
                    results.append(
                        'Best {}: {}/{}'.format(self.metric, np.argmin(
                            self.metric_hist) + 1, self.fmt.format(np.min(self.metric_hist)))
                    )
            self.progress_bar.set_postfix_str(', '.join(results))
        if self.total_iteration - 1 == env.iteration - env.begin_iteration:
            self.progress_bar.close()
            del self.progress_bar
            self.progress_bar = None


try:
    import xgboost as xgb

    class XGBFitProgressbar(xgb.callback.TrainingCallback):
        def __init__(self, n_estimators, precision=5, start_position=0, metric=None, greater_is_better=True, update_cycle=30):
            self.start_position = start_position
            self.n_estimators = n_estimators
            self.fmt = '{:.' + str(precision) + 'f}'
            self.metric = metric
            self.metric_hist = []
            self.greater_is_better = greater_is_better
            self.progress_bar = None
            self.update_cycle = update_cycle

        def __repr__(self):
            return 'XGBFitProgressbar'

        def before_training(self, model):
            self.progress_bar = tqdm(
                total=self.n_estimators, desc='Round', position=self.start_position, leave=False)
            self.prog = 0
            return model

        def after_iteration(self, model, epoch, evals_log):
            # 진행 상태를 업데이트
            self.prog += 1
            if (self.prog % self.update_cycle) != 0:
                return False
            self.progress_bar.update(self.update_cycle)

            results = []
            for data_name, metrics in evals_log.items():
                for eval_name, eval_results in metrics.items():
                    result = eval_results[-1]
                    results.append(
                        f'{data_name}_{eval_name}:{self.fmt.format(result)}')
                    if self.metric == f'{data_name}_{eval_name}':
                        self.metric_hist.append(result)

            if self.metric is not None and self.metric_hist:
                if self.greater_is_better:
                    best_round = np.argmax(self.metric_hist) + 1
                    best_value = np.max(self.metric_hist)
                else:
                    best_round = np.argmin(self.metric_hist) + 1
                    best_value = np.min(self.metric_hist)

                results.append(
                    f'Best {self.metric}: {best_round}/{self.fmt.format(best_value)}')

            self.progress_bar.set_postfix_str(', '.join(results))

            # False를 반환하면 학습이 계속 진행됨
            return False

        def after_training(self, model):
            # 학습이 종료되면 진행바를 닫음
            self.progress_bar.update(self.n_estimators)
            self.progress_bar.close()
            del self.progress_bar
            self.progress_bar = None
            return model
except:
    pass


class CatBoostFitProgressbar:
    def __init__(self, n_estimators, precision=5, start_position=0, metric=None, greater_is_better=True, update_cycle=10):
        self.start_position = start_position
        self.n_estimators = n_estimators
        self.fmt = '{:.' + str(precision) + 'f}'
        self.metric = metric
        self.metric_hist = list()
        self.greater_is_better = greater_is_better
        self.progress_bar = None
        self.update_cycle = update_cycle
        self.prog = 0

    def __repr__(self):
        return 'CatBoostFitProgressbar'

    def after_iteration(self, info):
        if self.progress_bar is None:
            self.progress_bar = tqdm(
                total=self.n_estimators, desc='Round', position=self.start_position, leave=False)

        self.prog += 1
        if (self.prog % self.update_cycle) != 0:
            return True
        self.progress_bar.update(self.update_cycle)
        results = list()
        if info.metrics is not None:
            for k, v in info.metrics.items():
                results_2 = list()
                for k2, v2 in v.items():
                    results_2.append('{}: {}'.format(
                        k2, self.fmt.format(v2[-1])))
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

            results.append(
                f'Best {self.metric}: {best_round}/{self.fmt.format(best_value)}')

        self.progress_bar.set_postfix_str(', '.join(results))
        if self.progress_bar.n == self.n_estimators:
            self.after_train()
        return True

    def after_train(self):
        if self.progress_bar is not None:
            self.progress_bar.close()
            del self.progress_bar
            self.progress_bar = None


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
        self.progress_bar = tqdm(
            total=n_splits, desc='Fold', position=self.start_position, leave=False)

    def end_fold(self, fold, train_metrics, valid_metrics, model_result_cv):
        self.progress_bar.update(1)
        results = list()
        if len(train_metrics) > 0:
            results.append(
                '{}±{}'.format(self.fmt.format(np.mean(train_metrics)),
                               self.fmt.format(np.std(train_metrics)))
            )
        results.append(
            '{}±{}'.format(self.fmt.format(np.mean(valid_metrics)),
                           self.fmt.format(np.std(valid_metrics)))
        )
        self.progress_bar.set_postfix_str(', '.join(results))

    def end(self):
        self.progress_bar.close()
        del self.progress_bar
        if clear_output is not None:
            clear_output()
        self.progress_bar = None


def cv_model(
        sp, model, model_params, df, X, y, predict_func, score_func, return_train_scores=True,
        result_proc=None, train_data_proc=None, train_params={}, groups=None,
        progress_callback=None
    ):
    train_scores, valid_scores = list(), list()
    valid_prds = list()
    model_result = list()
    sp_params = {'X': df[X], 'y': df[y], 'groups': None if groups is None else df[groups]}
    if progress_callback is None:
        progress_callback = BaseCallBack()
    progress_callback.start(sp.get_n_splits(**sp_params))
    for fold, (train_idx, valid_idx) in enumerate(sp.split(**sp_params)):
        progress_callback.start_fold(fold)
        df_cv_train, df_valid = df.iloc[train_idx], df.iloc[valid_idx]
        if train_data_proc != None:
            df_cv_train = train_data_proc(df_cv_train)
        valid_splitter = train_params.get("valid_splitter", 0)
        if valid_splitter == "oof":
            train_params = train_params.copy()
            train_params['valid'] = df_valid
        result = train_model(model, model_params, df_cv_train, X, y, **train_params)
        m = result['model']

        def predictor(x): return predict_func(m, x, X)
        result['predictor'] = predictor
        valid_prds.append(predictor(df_valid))
        if return_train_scores:
            train_scores.append(
                score_func(df_cv_train, predictor(df_cv_train))
            )
        valid_scores.append(score_func(df_valid, valid_prds[-1]))
        if result_proc is not None:
            if type(result_proc) is list:
                for proc in result_proc:
                    model_result.append(proc(result))
            else:
                model_result.append(result_proc(result))
        progress_callback.end_fold(
            fold, train_scores, valid_scores, model_result)
        del df_cv_train, df_valid, m
        result = None
        gc.collect()
    s_prd = pd.concat(valid_prds, axis=0).sort_index()
    progress_callback.end()
    ret = {'valid_scores': valid_scores,
           'valid_prd': s_prd, 'model_result': model_result}
    if return_train_scores:
        ret['train_scores'] = train_scores
    return ret


def cv(df, sp, hparams, config, adapter, use_gpu=False, **argv):
    if 'validation_splitter' in config:
        argv['validation_splitter'] = config.pop('validation_splitter')

    if 'train_data_proc' in config and 'train_data_proc_param' in hparams:
        config = config.copy()
        config['train_data_proc'] = partial(
            config['train_data_proc'], **hparams['train_data_proc_param'])

    ret = cv_model(
        sp=sp, df=df, **config, **adapter.adapt(hparams, is_train=False, use_gpu=use_gpu, **argv)
    )
    ret['hparams'] = hparams
    return ret


def train(df, hparams, config, adapter, use_gpu=False, **argv):
    hparam_ = adapter.adapt(hparams, is_train=True, use_gpu=use_gpu, **argv)
    train_params = hparam_.pop(
        'train_params') if 'train_params' in hparam_ else {}
    if 'train_data_proc' in config:
        data_proc = partial(config['train_data_proc'],
                            **hparams.get('train_data_proc_param', {}))
    else:
        def data_proc(x): return x
    result = train_model(df_train=data_proc(df), **hparam_, **config, **train_params), hparam_['X']
    if clear_output is not None:
        clear_output()
    return result


def save_predictor(path, model_name, adapter, objs, spec):
    model_filename = os.path.join(path, model_name + '.model')
    adapter.save_model(model_filename, objs['model'])
    joblib.dump(spec, os.path.join(path, model_name + '.spec'))


def load_predictor(path, model_name, adapter):
    model_filename = os.path.join(path, model_name + '.model')
    if os.path.exists(model_filename):
        spec = joblib.load(os.path.join(path, model_name + '.spec'))
        model = adapter.load_model(model_filename)
        return {'model': model, 'spec': spec}
    else:
        return None


def train_model(model, model_params, df_train, X, y, valid_splitter=None, fit_params={}, valid_config_proc=None, **argv):
    df_valid, X_valid = None, None
    result = {}
    if (valid_splitter is not None and valid_splitter != "oof") or (valid_splitter == "oof" and 'valid' in argv):
        if valid_splitter == "oof":
            df_valid = argv['valid']
        else:
            df_train, df_valid = valid_splitter(df_train)
        X_valid = df_valid[X]
    X_train = df_train[X]
    result['variables'] = X.copy()
    y_train = df_train[y]
    if df_valid is not None:
        y_valid = df_valid[y]
    if valid_config_proc is not None:
        if X_valid is not None:
            model_params_2, fit_params_2 = valid_config_proc(
                (X_train, y_train), (X_valid, y_valid)
            )
            result['valid_shape'] = X_valid.shape
        else:
            model_params_2, fit_params_2 = valid_config_proc(
                (X_train, y_train), None
            )
    else:
        model_params_2, fit_params_2 = {}, {}
    result['train_shape'] = X_train.shape
    result['target'] = y
    m = model(**model_params, **model_params_2)
    m.fit(X_train, y_train, **fit_params, **fit_params_2)
    del X_train, y_train
    if df_valid is not None:
        del X_valid, y_valid, df_valid
    gc.collect()
    result['model'] = m
    return result


class BaseAdapter():
    def save_model(self, filename, model):
        joblib.dump(model, filename)

    def load_model(self, filename):
        return joblib.load(filename)


class SklearnAdapter(BaseAdapter):
    def __init__(self, model):
        self.model = model

    def adapt(self, hparams, is_train=False, use_gpu=False, **argv):
        return {
            'model': self.model,
            'model_params': hparams.get('model_params', {}),
            'X': X,
            'result_proc': argv.get('result_proc', None)
        }

    def __str__(self):
        return str(self.model.__name__)


class LGBMAdapter(BaseAdapter):
    def __init__(self, model, progress=0):
        self.model = model
        self.callbacks = list()
        if progress > 0:
            self.callbacks.append(LGBMFitProgressbar(update_cycle=progress))

    def adapt(self, hparams, is_train=False, use_gpu=False, **argv):
        if use_gpu:
            hparams = hparams.copy()
            hparams['device'] = 'cuda'
        return {
            'model': self.model,
            'model_params': {'verbose': -1, **hparams['model_params']},
            'X': X,
            'train_params': {
                'fit_params': {
                    'categorical_feature': X_cat_feature,
                    'callbacks': self.callbacks
                },
            },
            'result_proc': argv.get('result_proc', lgb_learning_result),
        }


class XGBAdapter(BaseAdapter):
    def __init__(self, model, gpu='cuda', progress=0):
        self.model = model
        self.gpu = 'cuda'
        self.progress = progress

    def adapt(self, hparams, is_train=False, use_gpu=False, **argv):
        callbacks = list()
        if self.progress > 0:
            callbacks.append(
                XGBFitProgressbar(n_estimators=hparams['model_params'].get(
                    'n_estimators', 100), update_cycle=self.progress)
            )
        return {
            'model': self.model,
            'model_params': {
                **hparams.get('model_params', {}),
                'callbacks': callbacks,
                'device': self.gpu if use_gpu else 'cpu'
            },
            'X': X,
            'train_params': {
                'fit_params': {'verbose': False},
                'categorical_num': len(X_cat_feature)
            },
            'result_proc': argv.get('result_proc', xgb_learning_result),
        }


class CBAdapter(BaseAdapter):
    def __init__(self, model, gpu='GPU', progress=0):
        self.model = model
        self.gpu = gpu
        self.progress = progress

    def adapt(self, hparams, is_train=False, use_gpu=False, **argv):
        fit_params = {
            'callbacks': [
                CatBoostFitProgressbar(n_estimators=hparams['model_params'].get('n_estimators', 100), update_cycle=self.progress)
            ]
        }
        return {
            'model': self.model,
            'model_params': {
                **hparams['model_params'],
                'cat_features': X_cat_feature, 'verbose': False,
                'task_type': self.gpu if use_gpu else None
            },
            'X': X,
            'train_params': {
                'fit_params':  fit_params
            },
            'result_proc': argv.get('result_proc', cb_learning_result),
        }

    def save_model(self, filename, model):
        model.save_model(filename)

    def load_model(self, filename):
        model = self.model()
        return model.load_model(filename)
