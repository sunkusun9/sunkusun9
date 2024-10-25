from functools import partial

import pandas as pd
import numpy as np
import gc

import tensorflow  as tf
from tensorflow.keras.callbacks import Callback

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

def fit_tf_model_to_Y(
    model, X, Y, to_tf_dataset, loss, optimizer, epochs,
    sample_weights = None, transformer = None,
    verbose=0, metrics = None, batch_size=64, shuffle_size=10240000, eval_set=None, validation_splitter=None,
    cb = [], early_stopping = None, reduce_lr_on_plateau = None, lr_scheduler = None
):
    tf.keras.backend.clear_session()
    cb = cb.copy()
    if validation_splitter != None:
        if eval_set is not None:
            raise Exception("If validation_splitter is set, eval_set should not be set.")
        if sample_weights is None:
            X, X_ev, Y, Y_ev = validation_splitter(X, Y)
            eval_set = (X_ev, Y_ev, None)
        else:
            X, X_ev, Y, Y_ev, sample_weights, sample_weights_ev = validation_splitter(X, Y, sample_weights)
            eval_set = (X_ev, Y_ev, sample_weights_ev)
    if early_stopping is not None:
        cb.append(tf.keras.callbacks.EarlyStopping(**early_stopping))
    if reduce_lr_on_plateau is not None:
        cb.append(tf.keras.callbacks.ReduceLROnPlateau(**reduce_lr_on_plateau))
    if lr_scheduler is not None:
        cb.append(tf.keras.callbacks.LearningRateScheduler(**lr_scheduler))
    if transformer is not None:
        X = transformer.fit_transform(X, Y)
    ds_train = to_tf_dataset(X, Y, sample_weights)
    if eval_set is not None:
        if transformer is not None:
            ds_eval = to_tf_dataset(transformer.transform(eval_set[0]), eval_set[1], eval_set[2])
        else:
            ds_eval = to_tf_dataset(eval_set[0], eval_set[1], eval_set[2] if len(eval_set) > 2 else None)
    else:
        ds_eval = None
    if shuffle_size > 0:
        ds_train = ds_train.shuffle(shuffle_size)
    if batch_size > 0:
        ds_train = ds_train.batch(batch_size)
        if ds_eval is not None:
            ds_eval = ds_eval.batch(batch_size)
    optimizer_ = optimizer.__class__.from_config(optimizer.get_config())
    model.compile(loss=loss, optimizer=optimizer_,  metrics=metrics)
    history = model.fit(
        ds_train, epochs=epochs, 
        validation_data=ds_eval, verbose=verbose,
        callbacks=cb
    )
    tf.keras.backend.clear_session()
    del ds_train
    if ds_eval is not None:
        del ds_eval
    gc.collect()
    return history.history

def predict_tf_model(model, X, to_tf_dataset, batch_size=64, **argv):
    ds = to_tf_dataset(X).batch(batch_size)
    prd =  model.predict(ds, verbose=argv.get('verbose', 0))
    del ds
    gc.collect()
    return prd


def create_model(inp, o, config, embedding = None, 
                 l1=0, l2=0, proba = False,
                 cont_dtype=tf.float32, 
                 ord_dtype=tf.float32):
    if type(inp) == dict:
        X_cat, X_inp = list(), dict()
        for k, v in inp.items():
            if v[0] == 'cont':
                X_inp[k] = tf.keras.Input(dtype=cont_dtype, shape=(v[1], ), name=k)
                X_cat.append(X_inp[k])
            elif v[0] == 'ord':
                X_inp[k] = tf.keras.Input(dtype=ord_dtype, shape=(v[1], ), name=k)
                X_cat.append(X_inp[k])
            elif v[0] == 'emb':
                X_inp[k] =tf.keras.Input(dtype=tf.int32, shape=(v[3], ), name=k)
                if v[4] > 0 and v[5] > 0:
                    kreg = tf.keras.regularizers.L1L2(v[4], v[5])
                elif v[4] > 0:
                    kreg  = tf.keras.regularizers.L1(v[4])
                elif v[5] > 0:
                    kreg = tf.keras.regularizers.L2(v[5])
                else:
                    kreg = None
                X_emb = tf.keras.layers.Embedding(v[1], v[2], embeddings_regularizer = kreg,
                                                  dtype=cont_dtype, name=k+'_emb')(X_inp[k])
                X_emb = tf.keras.layers.Reshape((v[2] * v[3],), name=k + '_reshape_emb')(X_emb)
                X_cat.append(X_emb)
        X = tf.keras.layers.Concatenate(axis=-1, name='concat_inputs')(X_cat)
    else:
        X_inp = tf.keras.Input(dtype=cont_dtype, shape=(inp, ), name='input_layer')
        X = X_inp
    l_cnt = 0
    for i in config:
        l1 = i.get('l1', 0)
        l2 = i.get('l2', 0)
        if l1 > 0 and l2 > 0:
            kreg = tf.keras.regularizers.L1L2(l1, l2)
        elif l1 > 0:
            kreg = tf.keras.regularizers.L1(l1)
        elif l2 > 0:
            kreg = tf.keras.regularizers.L2(l2)
        else:
            kreg = None
        activation = i.get('activation', None)
        if activation in ['sigmoid', 'tanh']:
            kinit = tf.keras.initializers.GlorotNormal()
        else:
            kinit = tf.keras.initializers.HeNormal()
        X = tf.keras.layers.Dense(i['unit'], activation=activation, 
                                  kernel_initializer=kinit,
                                  kernel_regularizer=kreg,
                                  name='l_{}'.format(l_cnt))(X)
        bn = i.get('batch_norm', False)
        if bn:
            X = tf.keras.layers.BatchNormalization()(X)
        do = i.get('dropout', 0)
        if do > 0:
            X = tf.keras.layers.Dropout(do)(X)
        l_cnt += 1
    if l1 > 0 and l2 > 0:
        kreg = tf.keras.regularizers.L1L2(l1, l2)
    elif l1 > 0:
        kreg = tf.keras.regularizers.L1(l1)
    elif l2 > 0:
        kreg = tf.keras.regularizers.L2(l2)
    else:
        kreg = None
    X = tf.keras.layers.Dense(o, kernel_regularizer=kreg, name='l_output')(X)
    if proba:
        if o == 1:
            X = tf.keras.layers.Activation('sigmoid')(X)
        else:
            X = tf.keras.layers.Softmax()(X)
    return tf.keras.Model(inputs=X_inp, outputs=X)

def get_input(X, ordinal = None, embedding = None, **argv):
    inp = None
    if ordinal is not None or embedding is not None:
        inp = {}
        if type(X) == pd.DataFrame:
            df_cont = X.select_dtypes('float')
            if len(df_cont.columns) > 0:
                inp['X_cont'] =  ('cont', len(df_cont.columns))
            if ordinal is not None:
                inp['X_ord'] = ('ord', len(ordinal))
            if embedding is not None:
                for k, v in enumerate(embedding):
                    inp['X_emb_{}'.format(k)] = ('emb', v[1], v[2], len(v[0]), v[3], v[4])
        else:
            cont_len = X.shape[-1] - (0 if ordinal is None else ordinal)\
                                    - (0 if embedding is None else np.sum([i[0] for i in embedding]))
            if cont_len > 0:
                inp['X_cont'] = ('cont', cont_len)
            if ordinal is None:
                ordinal = 0
            if ordinal > 0:
                inp['X_ord'] = ('ord', ordinal)
            if embedding is not None:
                f_ = cont_len + ordinal
                for k, v in enumerate(embedding):
                    inp['X_emb_{}'.format(k)] = ('emb', v[1], v[2], v[0], v[3], v[4])
                    f_ += v[0]
    else:
        inp = X.shape[1]
    return inp
def create_dataset(X, y=None, sample_weights=None, ordinal=None, embedding=None):
    output_ = None
    if ordinal is not None or embedding is not None:
        arr = {}
        if type(X) == pd.DataFrame:
            df_cont = X.select_dtypes('float')
            if len(df_cont.columns) > 0:
                arr['X_cont'] =  df_cont
            if ordinal is not None:
                arr['X_ord'] = X[ordinal]
            if embedding is not None:
                for k, v in enumerate(embedding):
                    arr['X_emb_{}'.format(k)] = X[v[0]]
        else:
            cont_len = X.shape[-1] - (0 if ordinal is None else ordinal)\
                                    - (0 if embedding is None else np.sum([i[0] for i in embedding]))
            if cont_len > 0:
                arr['X_cont'] = X[:, :cont_len]
            if ordinal is None:
                ordinal = 0
            if ordinal > 0:
                arr['X_ord'] = X[:, cont_len: (cont_len + ordinal)]
            if embedding is not None:
                f_ = cont_len + ordinal
                for k, v in enumerate(embedding):
                    arr['X_emb_{}'.format(k)] = X[:, f_:f_+v[0]]
                    f_ += v[0]
        if type(y) == pd.Series:
            if sample_weights is None:
                return tf.data.Dataset.from_tensor_slices((arr, y.values))
            else:
                return tf.data.Dataset.from_tensor_slices((arr, y.values, sample_weights.values))
        else:
            if sample_weights is None:
                return tf.data.Dataset.from_tensor_slices((arr, y))
            else:
                return tf.data.Dataset.from_tensor_slices((arr, y, sample_weights))
    else:
        if type(X) == pd.DataFrame:
            if sample_weights is None:
                return tf.data.Dataset.from_tensor_slices((X.values, y.values))
            else:
                return tf.data.Dataset.from_tensor_slices((X.values, y.values, sample_weights.values))
        else:
            if sample_weights is None:
                return tf.data.Dataset.from_tensor_slices((X, y))
            else:
                return tf.data.Dataset.from_tensor_slices((X, y, sample_weights))

class FitProgressBar(Callback):
    def __init__(self, postfix_step = 10, precision = 5, metric=None, greater_is_better = True, start_position=0, prog_level=1):
        super().__init__()
        self.precision = precision
        self.fmt = '{:.' + str(self.precision) + 'f}'
        self.metric = metric
        self.metric_hist = list()
        self.greater_is_better = greater_is_better
        self.postfix_step = postfix_step
        self.start_position = 0
        self.step_progress_bar = None
        self.prog_level = prog_level
    
    def __repr__(self):
        return 'FitProgressBar'
        
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.epoch_progress_bar = tqdm(total=self.epochs, desc='Epoch', position=self.start_position, leave=False)
        self.step_progress_bar = None

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.steps = self.params['steps']
        if self.step_progress_bar is None and self.prog_level >= 1:
            self.step_progress_bar = tqdm(total=self.steps, desc=f"Step", position=self.start_position + 1, leave=False)
        elif self.step_progress_bar is not None:
            self.step_progress_bar.reset()

    def on_batch_end(self, batch, logs=None):
        if self.step_progress_bar is not None:
            self.step_progress_bar.update(1)
            if batch % self.postfix_step == 0:
                if logs is not None and self.step_progress_bar is not None:
                    postfix = {k: self.fmt.format(v) for k, v in logs.items()}
                    self.step_progress_bar.set_postfix(**postfix)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_progress_bar.update(1)
        if logs is not None:
            postfix = list()
            for k, v in logs.items():
                postfix.append('{}: {}'.format(k, self.fmt.format(v)))
                if k == self.metric:
                    self.metric_hist.append(v)
            if self.metric is not None:
                if self.greater_is_better:
                    postfix.append(
                        'Best {}: {}/{}'.format(self.metric, np.argmax(self.metric_hist) + 1, self.fmt.format(np.max(self.metric_hist)))
                    )
                else:
                    postfix.append(
                        'Best {}: {}/{}'.format(self.metric, np.argmin(self.metric_hist) + 1, self.fmt.format(np.min(self.metric_hist)))
                    )
            self.epoch_progress_bar.set_postfix_str(', '.join(postfix))
    def on_train_end(self, logs = None):
        if self.step_progress_bar is not None:
            self.step_progress_bar.close()
            del self.step_progress_bar
            self.step_progress_bar = None
        self.epoch_progress_bar.close()
        del self.epoch_progress_bar
        self.epoch_progress_bar = None

class NNEstimator(BaseEstimator):
    def __init__(
            self, optimizer = ('Adam', {'learning_rate': 0.001}), loss = None, model=None, model_params={}, 
            to_tf_dataset=None, random_state=None, transformer=None, 
            epochs=10, batch_size = 256, shuffle_size=102400,
            early_stopping = None, reduce_lr_on_plateau = None, lr_scheduler = None,
            validation_splitter = None
        ):
        """
    Neural Network Estimator with TensorFlow
    
    This class implements a neural network estimator using TensorFlow. It allows the user to specify
    the model configuration, optimizer, loss function, and various training parameters. The class is
    designed to integrate seamlessly with the scikit-learn framework and supports embedding layers,
    ordinal inputs, and continuous inputs.

    Args:
        optimizer (tuple or str or tf.keras.optimizers.Optimizer, optional): The optimizer to use.
            If a tuple is provided, it should be of the form (optimizer_name, config_dict).
            Default is ('Adam', {'learning_rate': 0.001}).
        loss (str or tf.keras.losses.Loss, optional): The loss function to use. Default is None.
        model (callable, optional): A callable that returns a TensorFlow model. If None, a model is created
            using the provided `model_params`. Default is None.
        model_params (dict, optional): The parameters for creating the model, such as layer configurations,
            embedding settings, and input types. If `model` is `None`, this dictionary is used to automatically
            create a default model with specified configurations. If not provided, the following default
            configurations are used:
            - config: list
                    The configuration of layers.
                    dict format: {'unit': unit size, 
                                'l1': l1 regularization, 
                                'l2': l2 regularization,
                                'batch_norm': batch normalization: True or False, 
                                'dropout': drop-out rate}
            - ordinal: list or integer
                    If ordinal is list, X should be DataFrame. In this case, the ordinal is the column names of ordinal inputs.
                    If ordinal is the number, ordinal is the size of ordinal inputs.
            - embedding: list
                    If X is DataFrame, the list is consisted of (list of columns, input dimension, output dimension, l1 regularization, l2 regularization) tuples
                    If X is numpy array, the list is consisted of (the size of input, input dimension, output dimension, l1 regularization, l2 regularization) tuples
        to_tf_dataset (callable, optional): A function to convert input data into a TensorFlow dataset.
            If None, a default function using `create_dataset` is used. Default is None.
        random_state (int, optional): Random seed for reproducibility. Default is None.
        transformer (callable, optional): A transformer to preprocess the input data before training.
            Default is None.
        epochs (int, optional): The number of epochs to train the model. Default is 10.
        batch_size (int, optional): The batch size for training. Default is 256.
        shuffle_size (int, optional): The size of the shuffle buffer for training data. Default is 102400.
        early_stopping (dict, optional): Parameters for early stopping callback. If None, early stopping is not used.
        reduce_lr_on_plateau (dict, optional): Parameters for reducing learning rate on plateau callback.
            If None, learning rate is not reduced on plateau.
        lr_scheduler (dict, optional): Parameters for learning rate scheduler callback. If None, learning rate is not scheduled.
        validation_splitter (callable, optional): A function to split the data into training and validation sets.
            If None, no validation split is performed.

    Examples:
        ```python
        from sklearn.preprocessing import RobustScaler, OrdinalEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        import pandas as pd
        import sgnn

        # Load and prepare the dataset
        df_abalone = pd.read_csv('data/abalone.data', header=None)
        df_abalone.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 
                              'Viscera_weight', 'Shell_weight', 'Rings']

        X_cont = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
        X_cat = ['Sex']
        X_all = X_cont + X_cat

        # Example 1: Creating a model with default configurations
        clf_nn = Pipeline([
            ('ct', ColumnTransformer([
                ('std', RobustScaler(quantile_range=(0.01, 0.95)), X_cont),
                ('ord', OrdinalEncoder(), X_cat)
            ])),
            ('nn', sgnn.NNClassifier(
                model_params={
                    'config': [
                        {'unit': 128, 'activation': 'relu', 'batch_norm': True},
                        {'unit': 128, 'activation': 'relu', 'batch_norm': True},
                        {'unit': 64, 'activation': 'relu', 'batch_norm': True},
                        {'unit': 32, 'activation': 'relu', 'batch_norm': True},
                    ],
                    'embedding': [(1, 3, 2, 0, 0)]
                },
                optimizer=('Adam', {'learning_rate': 0.001}),
                batch_size=128,
                shuffle_size=204800,
                validation_splitter=lambda X, y: train_test_split(X, y, test_size=0.1, stratify=y)),
                reduce_lr_on_plateau={'factor': 0.1, 'patience': 5},
                early_stopping={'patience': 10},
                epochs=150
            ))
        ])

        # Fit the model
        clf_nn.fit(
            df_abalone[X_all], df_abalone['Rings'] > df_abalone['Rings'].mean(), 
            nn__verbose=0,
            nn__cb=[sgnn.FitProgressBar(metric='val_Accuracy', greater_is_better=False, postfix_step=30, prog_level=1)],
            nn__metrics=['Accuracy'],
        )

        # Example 2: Creating a custom model using tf.keras.Model
        import tensorflow as tf

        reg_nn = Pipeline([
            ('ct', ColumnTransformer([
                ('std', RobustScaler(quantile_range=(0.01, 0.95)), X_cont),
                ('ord', OneHotEncoder(drop='first'), X_cat)
            ])),
            ('nn', sgnn.NNRegressor(
                model=tf.keras.Sequential,
                model_params={
                    'layers': [
                        tf.keras.Input(shape=(len(X_cont) + 2,)),
                        tf.keras.layers.Dense(8, activation='relu'),
                        tf.keras.layers.Dense(4, activation='relu'),
                        tf.keras.layers.Dense(1),
                    ]
                },
                to_tf_dataset=lambda X, y, w: tf.data.Dataset.from_tensor_slices((X, y)),
                optimizer=('Adam', {'learning_rate': 0.001}),
                batch_size=128,
                shuffle_size=204800,
                validation_splitter=lambda X, y: train_test_split(X, y, test_size=0.1, 
                                        stratify=pd.qcut(y, [0, 0.2, 0.4, 0.6, 0.8, 1.0])),
                reduce_lr_on_plateau={'factor': 0.1, 'patience': 5},
                early_stopping={'patience': 10},
                epochs=150
            ))
        ])

        # Fit the custom model
        reg_nn.fit(
            df_abalone[X_all], df_abalone['Rings'], 
            nn__verbose=0,
            nn__cb=[sgnn.FitProgressBar(metric='val_MSE', greater_is_better=False, postfix_step=30, prog_level=1)],
            nn__metrics=['MSE'],
        )
        ```
        """
        self.model = model
        self.loss = loss
        self.model_params = model_params
        if to_tf_dataset is not None:
            self.to_tf_dataset = to_tf_dataset
        else:
            self.to_tf_dataset = partial(create_dataset, embedding = model_params.get('embedding', None), ordinal = model_params.get('ordinal', None))
        self.random_state =  random_state
        self.optimizer = optimizer
        self.epochs = epochs
        self.transformer = transformer
        self.is_fitted_ = False
        self.batch_size = batch_size
        self.shuffle_size= shuffle_size
        self.early_stopping = early_stopping
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_scheduler = lr_scheduler
        self.random_state = random_state
        self.validation_splitter = validation_splitter
        self.model_ = None
    
    def fit(self, X, y=None, sample_weights=None, **argv):
        if 'refit' not in argv or not argv['refit'] or not self.is_fitted:
            if self.model_ is None:
                if self.model is None:
                    self.model_ = create_model(
                        inp = get_input(X, **self.model_params),
                        proba=self.classes_ is not None, 
                        o=1 if self.classes_ is None or self.is_binary else len(self.classes_),
                        **self.model_params
                    )
                else:
                    self.model_ = self.model(**self.model_params)
        if self.transformer is not None:
            self.transformer_ = clone(self.transformer)
        else:
            self.transformer_ = None
        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
        if self.loss_ is None:
            if self.loss_ is None:
                self.loss_ = tf.keras.losses.MeanSquaredError()
            elif type(self.loss_) == 'tuple':
                self.loss_ = tf.keras.losses.get({'class_name': self.loss_[0], 'config': self.loss_[1]})
            elif type(self.loss_) == 'str':
                self.loss_ = tf.keras.losses.get({'class_name': self.loss_})
        if type(self.optimizer) == tuple:
            self.optimizer_ = tf.keras.optimizers.get({'class_name': self.optimizer[0], 'config': self.optimizer[1]})
        self.history_ = fit_tf_model_to_Y(
            self.model_, X, y, 
            self.to_tf_dataset, self.loss_, self.optimizer_, self.epochs, 
            sample_weights=sample_weights,
            batch_size = self.batch_size,
            shuffle_size = self.shuffle_size,
            transformer = self.transformer_,
            early_stopping = self.early_stopping,
            reduce_lr_on_plateau = self.reduce_lr_on_plateau,
            lr_scheduler = self.lr_scheduler,
            validation_splitter = self.validation_splitter,
            **argv
        )
        if self.history_ is not None and len(self.history_) > 0:
            self.epochs_ = max([len(i) for i in self.history_.values()])
        self.is_fitted_ = True
        return self
    
    def model_summary(self):
        if not self.is_fitted_: 
            raise Exception('Not Fitted')
        self.model_.summary()
    
    def predict(self, X, **argv):
        if not self.is_fitted_: 
            raise Exception('Not Fitted')
        return predict_tf_model(self.model_, X, self.to_tf_dataset, batch_size=self.batch_size, **argv)

class NNClassifier(ClassifierMixin, NNEstimator):
    def fit(self, X, y, **argv):
        self.le_ = LabelEncoder()
        y_lbl = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.is_binary = len(self.classes_) == 2
        if self.loss is None:
            self.loss_ = tf.keras.losses.BinaryCrossentropy() if self.is_binary else tf.keras.losses.SparseCategoricalCrossentropy()
        else:
            self.loss_ = self.loss
        return super().fit(X, y_lbl, **argv)

    def predict(self, X, **argv):
        y_prd = super().predict(X, **argv)
        if self.is_binary:
            return self.le_.inverse_transform(
                np.where(np.squeeze(y_prd) > 0.5, 1, 0)
            )
        else:
            return self.le_.inverse_transform(np.argmax(
                y_prd, axis=-1
            ))
    
    def predict_proba(self, X, **argv):
        y_prd = super().predict(X, **argv)
        if self.is_binary:
            prob = np.squeeze(y_prd)
            return np.vstack([1 - prob, prob]).T
        else:
            return y_prd

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X))

class NNRegressor(NNEstimator, RegressorMixin):
    def fit(self, X, y, **argv):
        self.classes_ = None
        if self.loss is None:
            self.loss_ = tf.keras.losses.MeanSquaredError()
        else:
            self.loss_ = self.loss
        return super().fit(X, y, **argv)
    
    def predict(self, X, **argv):
        ds_, _ = create_dataset(X, ordinal=self.ordinal, 
                                embedding=self.embedding, batch_size=self.batch_size, shuffle_size=0)
        return np.squeeze(self.model_.predict(ds_, **argv))
    
    def score(self, X, y, sample_weight=None):
        return r2_score(y, self.predict(X))