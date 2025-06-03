from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit, StratifiedKFold
from sklearn.linear_model import LinearRegression
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



def scheduler(epoch, lr, decay_epochdecay_epoch = 10, total_epoch = 30):
    if epoch < decay_epoch:
        return lr
    else:
        return lr * np.cos((epoch -  decay_epoch - 1) / (total_epoch - decay_epoch) * np.pi)