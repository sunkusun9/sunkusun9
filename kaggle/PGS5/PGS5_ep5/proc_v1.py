from sklearn.pipeline import make_pipeline
import sgpp
import polars as pl

p = make_pipeline(
    sgpp.PolarsProcessor(),
    sgpp.ExprProcessor({
        'Sex': pl.col('Sex') == 'male',
        'Duration_log': pl.col('Duration').log(),
        'Heart_Rate_div_Weight_sqrt': pl.col('Heart_Rate') / pl.col('Weight').sqrt(),
        'Age_div_Weight_sqrt': pl.col('Age') / pl.col('Weight').sqrt(),
        'Body_Temp_div_Heart_Rate': pl.col('Body_Temp') / pl.col('Heart_Rate'),
        'Weight_2_Duration': pl.col('Body_Temp') ** 2 / pl.col('Duration')
    }),
    sgpp.PandasConverter(index_col = 'id')
)