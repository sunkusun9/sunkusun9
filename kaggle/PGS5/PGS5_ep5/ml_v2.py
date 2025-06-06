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

X_cat = ['Sex']
X_num = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
X_all = X_cat + X_num
target = 'Calories_Log'

kf = KFold(4, random_state = 123, shuffle = True)
skf = StratifiedKFold(5, random_state = 123, shuffle = True)
ss = ShuffleSplit(1, random_state = 123)

def get_validation_splitter(validation_fraction):
    return lambda x: train_test_split(x, test_size = validation_fraction)

config = {
    'predict_func': lambda m, df, X: pd.Series(m.predict(df[X]), index = df.index),
    'score_func': lambda df, prds: root_mean_squared_error(df[target], prds),
    'validation_splitter': get_validation_splitter,
    'progress_callback': sgml.ProgressCallBack(), 
    'return_train_scores': True,
    'y': target,
    'sp_y': 'duration_bin'
}

xgb_adapter = sgml.XGBAdapter(xgb.XGBRegressor)
lgb_adapter = sgml.LGBMAdapter(lgb.LGBMRegressor)
cb_adapter = sgml.CBAdapter(cb.CatBoostRegressor)
lr_adapter = sgml.SklearnAdapter(LinearRegression)
nn_adapter = sgnn.NNAdapter(sgnn.NNRegressor, progress = 100)

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.cos((epoch - 9) / 25 * np.pi)