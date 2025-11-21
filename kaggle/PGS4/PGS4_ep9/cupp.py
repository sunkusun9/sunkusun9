from sklearn.base import TransformerMixin
import pandas as pd

class PGS4EP9Processor(TransformerMixin):
    def __init__(self):
        self.fuel_pat = "(?P<fuel_type_engine>Gasoline Fuel|Flex Fuel Capability|Diesel Fuel|Gasoline\\/Mild Electric Hybrid|Gas\\/Electric Hybrid|Plug-In Electric\\/Gas|Electric Fuel System|[0-9]+V .*|DOHC Turbo|Turbo)"
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, **argv):
        return pd.concat([
            X['engine'].str.extract("(?P<HP>[0-9.]+)HP ").astype('float32'), 
            X['engine'].str.extract("(?P<displacement>[0-9.]+)L|Liter").astype('float32'), 
            X['engine'].str.extract("(?P<engine_type>[^ ]+ Cylinder Engine|Electric Motor| [VI][0-9]+ )")['engine_type'].fillna('Unknown').str.replace('Cylinder Engine', '').str.strip().astype('category'),
            X['engine'].str.extract(self.fuel_pat).fillna('Unknown').astype('category'),
            X['engine'].astype('category')
        ], axis=1)
    
    def get_params(self, deep=True):
        return {}

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return ['HP', 'displacement', 'engine_type', 'full_type_engine']                