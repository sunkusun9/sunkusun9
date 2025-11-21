from sklearn.pipeline import make_pipeline
import sgpp
import polars as pl
import numpy as np

from itertools import combinations, product

X_num = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
X_r = [i + '_r' for i in X_num]
X_o2 = [(i + '_2' if i == j else i + '_mul_'+ j, (i, j)) for i, j in product(X_num, X_num) if i <= j]
X_1d1 = [(i + '_div_' + j, (i, j)) for i, j in product(X_num, X_num) if i != j]
X_o2d1 = [
    (i + '_div_' + j, (i, j)) for (i, a), j in product(X_o2, X_num) if j not in a 
]
X_o1d2 = [
    (i + '_div_' + j, (i, j)) for i, (j, a) in product(X_num, X_o2) if i not in a 
]

X_o3 = [
    (
        i + '_3' if i == j and j == k else i + '_2_mul_' + k if i == j else i + '_2_mul_' + j if i == k  else j + '_2_mul_' + i if j == k else i + '_mul_' + j + '_mul_' + k,
        (i, j, k)
    )
    for i, j, k in product(X_num, X_num, X_num) if i <= j and i <= k and j <= k
]

X_log = [i + '_log' for i in X_num]
X_sqrt_d = [i + '_sqrt_d' for i in X_num]
X_sqrt = [
    (i + '_sqrt' if i == j else i + '_div_' + j + '_sqrt', (i, j + '_sqrt_d')) for i, j in product(X_num, X_num)
]

var_list = [(a + '_r', 1 / pl.col(a).cast(pl.Float32)) for a in X_num]
var_list.extend([(a + '_log', pl.col(a).cast(pl.Float32).log()) for a in X_num])
var_list.extend([(a + '_sqrt_d', 1 / pl.col(a).cast(pl.Float32).sqrt()) for a in X_num])
var_list.extend([(i, pl.col(a).cast(pl.Float32) * pl.col(b).cast(pl.Float32)) for i, (a, b) in X_o2])
var_list.extend([(i, pl.col(a).cast(pl.Float32) / pl.col(b).cast(pl.Float32)) for i, (a, b) in X_1d1])
var_list.extend([(i, pl.col(a).cast(pl.Float32) * pl.col(b).cast(pl.Float32) * pl.col(c).cast(pl.Float32)) for i, (a, b, c) in X_o3])

var_list2 = [(i, pl.col(a) / pl.col(b)) for i, (a, b) in X_o2d1 + X_o1d2]
var_list2.extend([(i, pl.col(a) * pl.col(b)) for i, (a, b) in X_sqrt])

p = make_pipeline(
    sgpp.PolarsProcessor(),
    sgpp.ExprProcessor({
        **{'Sex': pl.col('Sex') == 'male', 'const': 1, 'duration_bin': pl.col('Duration').qcut(10, labels = [str(i) for i in np.arange(10)]).fill_null("0")},
        **{k: v for k, v in var_list}
    }),
    sgpp.ExprProcessor({
        **{k: v for k, v in var_list2}
    }),
    sgpp.PandasConverter(index_col = 'id')
)
