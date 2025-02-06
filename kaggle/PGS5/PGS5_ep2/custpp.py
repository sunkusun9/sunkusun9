from sklearn.base import TransformerMixin

class WeightCapacityProcessor(TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y = None):
        self.median_ = X['Weight Capacity (kg)'].median()
        self.vars_ = X.columns.tolist()
        return self
    def transform(self, X):
        return X.assign(
            wc_i = lambda x: x['Weight Capacity (kg)'].fillna(self.median_).round().astype('int').astype('category'),
            wc_i2 = lambda x: x['Weight Capacity (kg)'].fillna(self.median_).astype('category')
        )
    def get_params(self, deep=True):
        return {
        }

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return self.vars_ + ['wc_i', 'wc_i2']