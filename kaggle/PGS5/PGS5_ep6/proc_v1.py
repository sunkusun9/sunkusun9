import sgpp, sgutil
import polars as pl
from sklearn.pipeline import make_pipeline

p = make_pipeline(
    sgpp.PolarsProcessor(),
    sgpp.ColumnNameCleaner(),
    sgpp.PandasConverter('id'),
    sgpp.ApplyWrapper(
        sgpp.CatCombiner([(['Crop_Type', 'Soil_Type'], 'Crop_Soil')]), ['Crop_Type', 'Soil_Type']
    )
)
target = 'Fertilizer_Name'
X_cat = ['Soil_Type', 'Crop_Type']
X_cat2 = ['Crop_Soil']
X_num = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
X_all = X_cat + X_num + X_cat2

sc = sgutil.SGCache('img', 'result', 'model')