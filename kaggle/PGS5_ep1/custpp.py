import pandas as pd
import numpy as np
import holidays

from sklearn.base import TransformerMixin

class CountryEncoder(TransformerMixin):
    def __init__(self, gdp_file = 'gdp.csv'):
        self.gdp_file = gdp_file
        

    def fit(self, X, y = None):
        df_gdp = pd.read_csv(self.gdp_file)
        s_country = pd.Series(
                dict(zip(np.sort(X['country'].unique()), ["CA", "FI", "IT", "KE", "NO", "SG"]))
        )
        self.s_holiday = pd.concat([
            pd.Series(b, name = a).to_frame().unstack()
            for a, b in s_country.map(lambda x: holidays.country_holidays(x, years = range(2010, 2020))).items()
        ]).rename('holiday')
        
        self.s_gdp = df_gdp.loc[
            df_gdp.iloc[:, 0].isin(X['country'].unique())
        ].drop(columns = ['Code']).set_index('Country Name').loc[:, '2010':'2019'].rename(columns=lambda x: int(x)).stack().rename('gdp')
        self.features = X.columns.tolist()
        return self

    def transform(self, X):
        return X.assign(
            date = lambda x: x['date'].dt.date
        ).join(
            self.s_holiday, on = ['country', 'date']
        ).join(
            self.s_gdp, on = ['country', 'year'] 
        ).assign(
            holiday = lambda x: x['holiday'].fillna('None').astype('category')
        )

    def get_params(self, deep = True):
        return {}
        
    def set_output(self, transform = 'pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return self.features + ['holiday', 'gdp']

class RatioEncoder(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        self.s_ratio_ = pd.concat([
                X[['year', 'md', 'product']], y
            ], axis = 1
        ).assign(
            y_2 = lambda x: x['year'] % 2,
            ratio = lambda x: x.groupby(['year', 'md', 'product'], observed=False)['num_sold'].transform('sum')\
                             / x.groupby(['year', 'md'], observed=False)['num_sold'].transform('sum')
        ).groupby(['y_2', 'md', 'product'], observed=False)['ratio'].mean()
        self.features_ = X.columns.tolist() + ['product_ratio']
        return self

    def transform(self, X):
        return pd.concat([
            X, X[['year', 'md', 'product']].assign(
                y_2 = lambda x: x['year'] % 2
            )[['y_2', 'md', 'product']].apply(lambda x: tuple(x), axis=1).map(
                self.s_ratio_
            ).rename('product_ratio')
        ], axis = 1)
    
    def get_params(self, deep = True):
        return {}
        
    def set_output(self, transform = 'pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return self.features_

    def __str__(self):
        return "RatioEncoder"