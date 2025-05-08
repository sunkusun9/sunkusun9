from sklearn.pipeline import make_pipeline
import sgpp
import polars as pl

p = make_pipeline(
    sgpp.PolarsProcessor(),
    sgpp.ExprProcessor({'Sex': pl.col('Sex') == 'male'}),
    sgpp.PandasConverter(index_col = 'id')
)