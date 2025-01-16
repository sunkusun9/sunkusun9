import pandas as pd
import dproc
from sklearn.base import TransformerMixin


class CatNomTransformer(TransformerMixin):
    def __init__(self, mapper):
        self.mapper = mapper

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return pd.DataFrame(
            pd.concat([
                dproc.replace_cat(X.loc[X[k].isin(v.keys()), k].astype('category'), v).rename(k)
                for k, v in self.mapper.items()
            ], axis=1), index = X.index
        )

    def get_params(self, deep=True):
        return {
            "mapper": self.mapper
        }

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return list(self.mapper)

class NumProcessor(TransformerMixin):
    X_1 = ['Curricular units 2nd sem (approved)', 'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (enrolled)', 'Curricular units 1st sem (enrolled)',
           'Curricular units 2nd sem (evaluations)', 'Curricular units 1st sem (evaluations)', 'Curricular units 2nd sem (credited)', 'Curricular units 1st sem (credited)']
    X_2 = ['Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (without evaluations)']
    def __init__(self):
        pass
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        return pd.concat([
            X[NumProcessor.X_1].clip(0, 14),
            X[NumProcessor.X_2].clip(0, 8),
            X['Age at enrollment'].clip(0, 30),
            (X['Curricular units 1st sem (grade)'].round().astype('int').clip(0, 16).replace({1: 0, 2: 0}) * (X['Curricular units 1st sem (evaluations)'] != 0)).rename('Curricular units 1st sem (grade)'),
            (X['Curricular units 2nd sem (grade)'].round().astype('int').clip(0, 16).replace({1: 0, 2: 0}) * (X['Curricular units 2nd sem (evaluations)'] != 0)).rename('Curricular units 2nd sem (grade)'),
        ], axis = 1).rename(columns = lambda x: 'np_' + x)
    def get_params(self, deep=True):
        return {}

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return ['np_{}'.format(i) 
                for i in  NumProcessor.X_1 + NumProcessor.X_2 + ['Age at enrollment', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']]

class NumCombiner(TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        return pd.concat([
            dproc.combine_cat(
                pd.concat([
                    X['np_Curricular units 2nd sem (evaluations)'].astype('category'), X['np_Curricular units 2nd sem (grade)'].astype('category')
                ], axis=1), '_'
            ).rename('2nd_eval_grade'),
            dproc.combine_cat(
                pd.concat([
                    X['np_Curricular units 1st sem (evaluations)'].astype('category'), X['np_Curricular units 1st sem (grade)'].astype('category')
                ], axis=1), '_'
            ).rename('1st_eval_grade'),
            dproc.combine_cat(
                pd.concat([
                    X['np_Curricular units 1st sem (approved)'].astype('category'), X['np_Curricular units 2nd sem (approved)'].astype('category')
                ], axis=1), '_'
            ).rename('approved'),
        ], axis = 1)
    def get_params(self, deep=True):
        return {}

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return ['2nd_eval_grade', '1st_eval_grade', 'approved']