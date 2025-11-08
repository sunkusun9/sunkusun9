from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import sgml
import pandas as pd
import numpy as np
import sgnn

def mapk_prob(y_true, y_prob, k=3):
    return (
        # Step 1: argsort(-y_prob) gives indices of predicted classes in descending order of probability
        # Step 2: Compare top-k predictions with the true label (broadcasted across k predictions)
        # Step 3: Assign precision weights: 1/1, 1/2, ..., 1/k for top-k ranks
        # Step 4: Take the dot product to get the weighted score for each observation
        # Step 5: Average over all observations
        (np.argsort(-y_prob, axis=1) == np.expand_dims(y_true, axis=-1))[:, :k].dot(1 / np.arange(1, k + 1))
    ).mean()



def scheduler(epoch, lr, decay_epoch = 10, total_epoch = 30):
    if epoch < decay_epoch:
        return lr
    else:
        return lr * np.cos((epoch -  decay_epoch - 1) / (total_epoch - decay_epoch) * np.pi)

def get_validation_splitter(validation_fraction):
    return lambda x: train_test_split(x, test_size = validation_fraction, stratify = x[target + '_l'])

def train_data_proc(x, org = None):
    if org is not None:
        return pd.concat([x, org], axis = 0)
    else:
        return x

ss = StratifiedShuffleSplit(1, random_state = 123)
skf = StratifiedKFold(5, random_state = 123, shuffle = True)

config = {
    'predict_func': lambda m, df, X: pd.DataFrame(m.predict_proba(df[X]), index = df.index),
    'score_func': lambda df, prds: mapk_prob(df['Fertilizer_Name_l'], prds),
    'validation_splitter': get_validation_splitter,
    'progress_callback': sgml.ProgressCallBack(), 
    'return_train_scores': True,
    'train_data_proc': train_data_proc,
    'y': 'Fertilizer_Name_l',
}

xgb_adapter = sgml.XGBAdapter(xgb.XGBClassifier, progress = 50)
lgb_adapter = sgml.LGBMAdapter(lgb.LGBMClassifier, progress = 50)
cb_adapter = sgml.CBAdapter(cb.CatBoostClassifier, progress = 50)
nn_adapter = sgnn.NNAdapter(sgnn.NNClassifier, progress = 100)
lr_adapter = sgml.SklearnAdapter(LogisticRegression)

