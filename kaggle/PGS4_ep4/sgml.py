from sklearn.base import clone
from sklearn.pipeline import make_pipeline
import pickle as pkl
import numpy as np
import pandas as pd
import gc


def gb_valid_config(train_set, valid_set):
    return {}, {'eval_set': [train_set, valid_set] if valid_set is not None else [train_set]}

def sgnn_valid_config(train_set, valid_set):
    return {}, {'eval_set': valid_set if valid_set is not None else None}

def lgb_learning_result(m, train_result):
    return (
        pd.concat([
            pd.DataFrame(
                m.evals_result_[i]
            ).rename(columns=lambda x: (i, x)) for i in m.evals_result_.keys()
        ], axis=1).pipe(
            lambda x: x.reindex(columns = pd.MultiIndex.from_tuples(x.columns.tolist(), names=['set', 'metric'])).swaplevel(axis=1)
        ) if len(m.evals_result_) > 0 else None, 
        pd.Series(
            m.feature_importances_, index=train_result['variables']
        ).sort_values(),
        train_result
    )

def xgb_learning_result(m, train_result):
    return (
        pd.concat([
            pd.DataFrame(
                m.evals_result_[i]
            ).rename(columns=lambda x: (i, x)) for i in m.evals_result_.keys()
        ], axis=1).pipe(
            lambda x: x.reindex(columns = pd.MultiIndex.from_tuples(x.columns.tolist(), names=['set', 'metric'])).swaplevel(axis=1)
        ), 
        pd.Series(
            m.feature_importances_, index=train_result['variables']
        ).sort_values(),
        train_result
    )

def cb_learning_result(m, train_result):
    return (
        pd.concat([
            pd.DataFrame(
                m.evals_result_[i]
            ).rename(columns=lambda x: (i, x)) for i in m.evals_result_.keys()
        ], axis=1).pipe(
            lambda x: x.reindex(columns = pd.MultiIndex.from_tuples(x.columns.tolist(), names=['set', 'metric'])).swaplevel(axis=1)
        ), 
        pd.Series(
            m.feature_importances_, index=train_result['variables']
        ).sort_values(),
        train_result
    )

def sgnn_learning_result(m, train_result):
    return (
        pd.DataFrame(m.history_),
        train_result
    )

def train_model(model, model_params, df_train, X, y, valid_splitter=None, preprocessor=None, fit_params={}, valid_config_proc = None):
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
        object, dict
        model instance, train result
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
    if valid_config_proc is not None:
        if X_valid is not None:
            model_params_2, fit_params_2 = valid_config_proc((X_train, df_train[y]), (X_valid, df_valid[y]))
            result['valid_shape'] = X_valid.shape
        else:
            model_params_2, fit_params_2 = valid_config_proc((X_train, df_train[y]), None)
    else:
        model_params_2, fit_params_2 = {}, {}
    result['train_shape'] = X_train.shape
    result['target'] = y
    m =  model(**model_params, **model_params_2)
    m.fit(X_train, df_train[y], **fit_params, **fit_params_2)
    del X_train
    if df_valid is not None:
        del X_valid, df_valid
    gc.collect()
    if preprocessor is not None:
        m = make_pipeline(preprocessor, m)
    return m, result

def cv_model(sp, model, model_params, df, X, y, predict_func, eval_metric,
                   preprocessor=None, result_proc=None, train_data_proc=None, train_params={}, sp_y=None):
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
        eval_metric: function
            score functiongb_valid_config
        preprocessor: sklearn.preprocessing. 
            preprocessor. it will be connected using make_pipeline
        fit_params: dict
            parameters for fit
        sp_y: str
            splitter y value name
    Returns
        list, list, Series, list
        train_metrics, valid_metrics, s_prd, model_result_cv
    Example
    >>> X_cont =['Diameter', 'Whole weight.2', 'Whole weight.1', 'Shell weight', 'Length', 'Height_n', 'Whole weight']
    >>> X_cat = ['Sex']
    >>> X_all = X_cont + X_cat
    >>> def predict(m, df_valid, X):
    >>>     return pd.Series(m.predict(df_valid[X]), index=df_valid.index)
    >>> def eval_metric(y_true, prds):
    >>>     return mean_squared_error(y_true.sort_index(), prds.sort_index()) ** 0.5
    >>> def gb_valid_config(train_set, valid_set):
    >>>     return {}, {'eval_set': [train_set, valid_set] if valid_set is not None else [train_set]}
    >>> cv_model(StratifiedKFold(n_splits=5, random_state=123, shuffle=True), 
    >>>          lgb.LGBMRegressor, {'verbose': -1}, df_train_sp, X_all, 'target',
    >>>         predict_func=predict, eval_metric = eval_metric,
    >>>         train_params={
    >>>             'valid_splitter': lambda x: train_test_split(x, train_size=0.9, stratify=x['Rings'], random_state=123),
    >>>             'fit_params': {'categorical_feature': ['Sex'], 'callbacks': [lgb.early_stopping(5, verbose=False)]},
    >>>             'valid_config_proc': sgml.gb_valid_config
    >>>         }, sp_y = 'Rings'
    >>> )
    """
    train_metrics, valid_metrics = list(), list()
    valid_prds = list()
    if sp_y is None:
        sp_y = y
    model_result_cv = list()
    for train_idx, valid_idx in sp.split(df[X], df[sp_y]):
        df_cv_train, df_valid = df.iloc[train_idx], df.iloc[valid_idx]
        if train_data_proc != None:
            df_cv_train = train_data_proc(df_cv_train)
        m, train_result = train_model(model, model_params, df_cv_train, X, y, preprocessor=preprocessor, **train_params)
        valid_prds.append(predict_func(m, df_valid, X))
        valid_metrics.append(eval_metric(df_valid, valid_prds[-1]))
        train_metrics.append(eval_metric(df_cv_train, predict_func(m, df_cv_train, X)))
        if result_proc is not None:
            if preprocessor is None:
                train_result = result_proc(m, train_result)
            else:
                train_result = result_proc(m[-1], train_result)
        model_result_cv.append(train_result)
    s_prd = pd.concat(valid_prds, axis=0)
    return train_metrics, valid_metrics, s_prd, model_result_cv

class SGStacking:
    def __init__(self, sp, predict_func, eval_metric, greater_better=True):
        """
        Parameters
            sp: Splitter
                sklearn spliiter object
            predict_func: function
                extract prediction result from model
            eval_metric: function
                evaluation metric function
            greater_better: Boolean
                True: the greater, the better False: the greater, the worse
        Examples
        >>> cv5 = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
        >>> def predict(m, df_valid, X):
        >>>     return pd.Series(m.predict(df_valid[X]), index=df_valid.index)
        >>> def eval_metric(y_true, prds):
        >>>     return mean_squared_error(y_true.sort_index(), prds.sort_index()) ** 0.5
        >>> stk = SGStacking(sp=cv5, predict_func=predict, eval_metric=eval_metric,  greater_better=False)
        """
        self.sp = sp
        self.predict_func = predict_func
        self.eval_metric = eval_metric
        self.model_result = {}
        self.selected_models = {}
        self.greater_better = greater_better
        self.meta_model = None
        self.meta_X = None

    def get_result(self, model_name, model, model_param, X):
        """
        get the results of the model
        Parameters:
            model_name: str
                model name
            model: Class
                Class of model
            model_param: dict
                model parameter
            X: list
                list of variable names
        Returns:
            dict
            {
                'model': model list
                'model param': the parameters of model
                'train_metrics': the metrics of train
                'valid_metrics': the metrics of validation
            }
        """
        if model_name in self.model_result:
            result_ = self.model_result[model_name]
            model_key = str(model) + str(model_param) + ','.join(X)
            if model_key in result_['model_key']:
                idx = result_['model_key'].index(model_key)
                return {
                    'model': result_['model'][idx],
                    'model_param': result_['model_params'][idx],
                    'train_metrics': result_['train_metrics'][idx],
                    'valid_metrics': result_['valid_metrics'][idx],
                }
            return None
        return None
    
    def _put_result(self, model_name, model, model_params, X, train_metrics, valid_metrics, s_prd, model_result_cv):
        model_key = str(model) + str(model_params) + ','.join(X)
        if model_name in self.model_result:
            result_ = self.model_result[model_name]
            if model_key in result_['model_key']:
                idx = result_['model_key'].index(model_key)
                result_['train_metrics'][idx] = train_metrics
                result_['valid_metrics'][idx] = valid_metrics
                metric = np.mean(valid_metrics)
                if result_['best_result'] is None or \
                    (self.greater_better and metric >= np.max(result_['metric'])) or \
                    (not self.greater_better and metric <= np.min(result_['metric'])):
                    result_['best_result'] = (s_prd, model_result_cv)
                    if model_name in self.selected_models:
                        del self.selected_models[model_name]
                result_['metric'][idx] = metric
        else:
            result_ = {
                'model_key': [],
                'model': [],
                'model_params': [],
                'X': [],
                'train_metrics': [],
                'valid_metrics': [],
                'metric': [],
                'best_result': None
            }
            self.model_result[model_name] = result_
        result_['model_key'].append(model_key)
        result_['model'].append(model)
        result_['model_params'].append(model_params)
        result_['X'].append(X)
        result_['train_metrics'].append(train_metrics)
        result_['valid_metrics'].append(valid_metrics)
        metric = np.mean(valid_metrics)
        if result_['best_result'] is None or \
            (self.greater_better and metric >= np.max(result_['metric'])) or \
            (not self.greater_better and metric <= np.min(result_['metric'])):
            result_['best_result'] = (s_prd, model_result_cv)
            if model_name in self.selected_models:
                del self.selected_models[model_name]
        result_['metric'].append(metric)
    
    def compact_result(self, model_name):
        """
        Store only the result of best trial
        """
        result_new_ = {
            'model_key': [],
            'model': [],
            'model_params': [],
            'X': [],
            'train_metrics': [],
            'valid_metrics': [],
            'metric': [],
        }
        result_ = self.model_result[model_name]
        if self.greater_better:
            idx = np.argmax(result_['metric'])
        else:
            idx = np.argmin(result_['metric'])
        result_new_['model'].append(result_['model'][idx])
        result_new_['model_params'].append(result_['model_params'][idx])
        result_new_['X'].append(result_['X'][idx])
        result_new_['train_metrics'].append(result_['train_metrics'][idx])
        result_new_['valid_metrics'].append(result_['valid_metrics'][idx])
        result_new_['metric'].append(result_['metric'][idx])
        result_new_['best_result'] = result_['best_result']
        self.model_result[model_name] = result_new_

    def reset_model(self, model_name):
        del self.model_result[model_name]
    
    def get_best_results(self, model_names):
        tmp = list()
        for model_name in model_names:
            result_ = self.model_result[model_name]
            result_new_ = dict()
            if self.greater_better:
                idx = np.argmax(result_['metric'])
            else:
                idx = np.argmin(result_['metric'])
            result_new_['model'] = result_['model'][idx]
            result_new_['model_params'] = result_['model_params'][idx]
            result_new_['X'] = result_['X'][idx]
            result_new_['train_metrics'] = result_['train_metrics'][idx]
            result_new_['valid_metrics'] = result_['valid_metrics'][idx]
            tmp.append(pd.Series(result_new_))
        return pd.DataFrame(tmp).assign(
            model = lambda x: x['model'].apply(lambda x: str(x).split('.')[-1][:-2]),
            X = lambda x: x['X'].apply(lambda x: ','.join(x)),
            train_metrics = lambda x: x['train_metrics'].apply(lambda x: '{:.5f}±{:.5f}'.format(np.mean(x), np.std(x))),
            valid_metrics = lambda x: x['valid_metrics'].apply(lambda x: '{:.5f}±{:.5f}'.format(np.mean(x), np.std(x))),
        )
    
    def eval_model(self, model_name, model, model_params, df, X, y, 
                   preprocessor=None, result_proc=None, train_data_proc=None, train_params={}, sp_y=None, rerun=False):
        """
        Evaluate a base model and store the result.
        Parameters:
            model_name: str
                모델명
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
            preprocessor: sklearn.preprocessing. 
                preprocessor. it will be connected using make_pipeline
            result_proc: function
                the processor for the result of training 
            train_data_proc: function
                the processor for traing data
            train_params: 
                the parameter for train_model
            sp_y: str
                splitter y value name
            rerun: Boolean
                Rerun all the 
        Returns
            object, dict
            model instance, train result 
        Example
        >>> lgb_result, train_result = stk.eval_model(
        >>>    'lgb_1', lgb.LGBMRegressor, {'verbose': -1, 'n_estimators': 140}, dataset['train'], X_all, target,
        >>>    result_proc=lgb_learning_result,
        >>>    train_params={
        >>>         'valid_splitter': valid_splitter, 
        >>>         'fit_params': {'categorical_feature': ['Sex'], 'callbacks': [lgb.early_stopping(5, verbose=False)]}, 
        >>>         'valid_config_proc': gb_valid_config
        >>>     }, sp_y = 'Rings'
        >>> )
        """
        if not rerun:
            result = self.get_result(model_name, model, model_params, X)
            if result != None:
                return result, None
        train_metrics, valid_metrics, s_prd, model_result_cv = \
            cv_model(
                self.sp, model, model_params, df, X, y, self.predict_func, self.eval_metric,
                preprocessor=preprocessor, result_proc=result_proc, train_data_proc=train_data_proc, train_params=train_params, sp_y=sp_y
            )
        self._put_result(model_name, model, model_params, X, train_metrics, valid_metrics, s_prd, model_result_cv)
        return self.get_result(model_name, model, model_params, X), model_result_cv

    def get_model_results(self, model_name):
        """
        get the training results of the model
        Parameters:
            model_name: str
        Returns:
            DataFrame
            the dataframe which contains the training results
        """
        tmp = self.model_result[model_name].copy()
        del tmp['best_result'], tmp['model_key'], tmp['metric']
        return pd.DataFrame(tmp).assign(
            model = lambda x: x['model'].apply(lambda x: str(x).split('.')[-1][:-2]),
            X = lambda x: x['X'].apply(lambda x: ','.join(x)),
            train_metrics = lambda x: x['train_metrics'].apply(lambda x: '{:.5f}±{:.5f}'.format(np.mean(x), np.std(x))),
            valid_metrics = lambda x: x['valid_metrics'].apply(lambda x: '{:.5f}±{:.5f}'.format(np.mean(x), np.std(x))),
        )
        
    def select_model(self, model_name, df, X, y, 
                   preprocessor=None, result_proc=None, train_data_proc=None, train_params={}, rerun=False):
        """
        Select the model and fit the model with the best parameter for base model. And store the model instance and cv prediction of the model.
        Parameters:
            model_name: str
                모델명
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
            preprocessor: sklearn.preprocessing. 
                preprocessor. it will be connected using make_pipeline
            result_proc: function
                the processor for the result of training 
            train_data_proc: function
                the processor for traing data
            train_params: 
                the parameter for train_model
            rerun: Boolean
                Rerun all the 
        Returns:
            object, dict, float
            model instance, train result, train metric
        
        Example:
        >>> lgb_result, train_result = stk.eval_model(
        >>>     'lgb_1', lgb.LGBMRegressor, 
        >>>     {'verbose': -1, 'n_estimators': 1500, 'learning_rate': 0.01, 'colsample_bytree': 0.75, 'subsamples': 0.75, 'num_leaves': 63}, 
        >>>     dataset['train'], X_all, target,
        >>>     result_proc=lgb_learning_result,
        >>>     train_data_proc=partial(merge_org, df_org=df_org),
        >>>     train_params={
        >>>         'valid_splitter': valid_splitter, 
        >>>         'fit_params': {'categorical_feature': ['Sex'], 'callbacks': [lgb.early_stopping(5, verbose=False)]}, 
        >>>         'valid_config_proc': gb_valid_config
        >>>     }, sp_y = 'Rings'
        >>> )    
        """
        if not rerun and model_name in self.selected_models:
            return self.selected_models[model_name][0], self.selected_models[model_name][1], self.selected_models[model_name][2]
        result_ = self.model_result[model_name]
        if self.greater_better:
            idx = np.argmax(result_['metric'])
        else:
            idx = np.argmin(result_['metric'])
        model = result_['model'][idx]
        model_params = result_['model_params'][idx]
        if train_data_proc is not None:
            df = train_data_proc(df)
        m, train_result = train_model(model, model_params, df, X, y, preprocessor=preprocessor, **train_params)
        train_metric = self.eval_metric(df, self.predict_func(m, df, X))
        if result_proc is not None:
            if preprocessor is None:
                train_result = result_proc(m, train_result)
            else:
                train_result = result_proc(m[-1], train_result)
        self.selected_models[model_name] = (
            m, X, train_result, train_metric
        )
        return m, train_result, train_metric
   
    def eval_meta_model(self, model, model_params, model_names, df, y, result_proc=None, train_params={}, sp_y=None):
        """
        Evaluate the meta model
        Parameters:
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
            preprocessor: sklearn.preprocessing. 
                preprocessor. it will be connected using make_pipeline
            result_proc: function
                the processor for the result of training 
            train_data_proc: function
                the processor for traing data
            train_params: 
                the parameter for train_model
            rerun: Boolean
                Rerun all the 
        Returns:
            object, dict, float
            model instance, train result, train metric
        """
        df = pd.concat([
                self.model_result[i]['best_result'][0].rename(i) for i in model_names
            ] + [df], axis=1).sort_index()
        train_metrics, valid_metrics, s_prd, model_result_cv = cv_model(
            self.sp, model, model_params, df, model_names, y, self.predict_func, self.eval_metric, 
            result_proc=result_proc, train_params=train_params, sp_y=sp_y
        )
        return train_metrics, valid_metrics, model_result_cv
    
    def fit(self, model, model_params, model_names, df, y, result_proc=None, train_params={}):
        """
        fit the meta model
        Parameters:
            model: Class
                the class of meta model
            model_params:
                the parameter of meta model
            model_names:
                the names of base model
            s_y: Series or DataFrame
                the target value(s)
            result_proc: function
                the function to extract the result
            train_params: dict
                the parameters for train
        Returns:
            dict
                train result
        """
        df = pd.concat([
                self.model_result[i]['best_result'][0].rename(i) for i in model_names
            ] + [df], axis=1).sort_index()
        m, train_result = train_model(model, model_params, df, model_names, y, **train_params)
        train_metric = self.eval_metric(df, self.predict_func(m, df, model_names))
        if result_proc is not None:
            train_result = result_proc(m, train_result)
        self.meta_model = m
        self.meta_X = model_names
        return train_result
    
    def predict(self, df):
        """
        predict the values
        Parameters:
            df: DataFrame
                input dataframe
        """
        prds = list()
        for m_ in self.meta_X:
            m, X, _, _ = self.selected_models[m_]
            prds.append(pd.Series(m.predict(df[X]), name=m_))
        return self.meta_model.predict(pd.concat(prds, axis=1))

    def save_model(self, file_name):
        """
        save this model
        Parameters:
            file_name: str
                file name
        """
        model_contents = {
            'splitter': self.sp,
            'predict_func': self.predict_func,
            'eval_metric': self.eval_metric,
            'model_result': self.model_result,
            'selected_models': self.selected_models,
            'greater_better': self.greater_better,
            'meta_model': self.meta_model,
            'meta_X': self.meta_X
        }
        with open(file_name, 'wb') as f:
            pkl.dump(model_contents, f)
    
    def load_model(file_name):
        """
        load the model
        Parameters:
            file_name: str
                file name
        """
        with open(file_name, 'rb') as f:
            model_contents = pkl.load(f)
        stk = SGStacking(model_contents['splitter'], model_contents['predict_func'], model_contents['eval_metric'], model_contents['greater_better'])
        stk.model_result = model_contents['model_result']
        stk.selected_models = model_contents['selected_models']
        stk.meta_model = model_contents['meta_model']
        stk.meta_X = model_contents['meta_X']
        return stk