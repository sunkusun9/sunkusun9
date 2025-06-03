import sgpp
from sklearn.pipeline import make_pipeline

p = make_pipeline(
    sgpp.PolarsProcessor(),
    sgpp.ColumnNameCleaner(),
    sgpp.PandasConverter('id'),
)

import polars as pl
target = 'Fertilizer_Name'
X_cat = [k for k, v in df_train.dtypes.items() if v == 'category' and k not in [target, 'id']]
X_num = [k for k, v in df_train.dtypes.items() if k not in X_cat and k not in [target, 'id']]
X_all = X_cat + X_num