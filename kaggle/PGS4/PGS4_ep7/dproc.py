import os
import pandas as pd
import numpy as np
import polars as pl
import polars.selectors as cs
from functools import partial
from itertools import product
from collections import OrderedDict
import dill

def get_type_df(df):
    """
    get data type of polars df
    Parameters:
        df: pl.DataFrame, pl.LazyFrame
            DataFrame to examine the proper data type
    Returns:
        pd.DataFrame
            DataFrame which contains the information about the types
    Examples:
        >>> df_type = pl.scan_csv('test.csv').pipe(get_type_df)
    """
    stat = [
        ('min', cs.numeric().min()),
        ('max', cs.numeric().max()),
        ('na', cs.all().is_null().sum()),
        ('count', cs.all().count()),
        ('n_unique', cs.all().n_unique()),
    ]
    df_type = df.select(
        **{k: pl.struct(v) for k, v in stat}
    ).melt().unnest('value').rename({'variable': 'stat'})\
    .melt(id_vars='stat', variable_name='feature')
    if type(df_type) == pl.LazyFrame:
        df_type = df_type.collect()
    if type(df) == pl.lazyframe.frame.LazyFrame:
        dtypes = df.collect_schema().dtypes()
        columns = df.collect_schema().names()
    else:
        dtypes = df.dtypes
        columns = df.columns
    df_type = df_type.pivot(index='feature', columns='stat', values='value')\
                .to_pandas().set_index('feature').join(
                    pd.Series([str(i) for i in dtypes], index=columns, name='dtype')
                )
    for i, mn, mx in [
        ('f32', np.finfo(np.float32).min, np.finfo(np.float32).max),
        ('i32', np.iinfo(np.int32).min, np.iinfo(np.int32).max),
        ('i16', np.iinfo(np.int16).min, np.iinfo(np.int16).max),
        ('i8', np.iinfo(np.int8).min, np.iinfo(np.int8).max)
    ]:
        df_type[i] = df_type.loc[df_type['dtype'] != 'String'].apply(
            lambda x: (x['min'] >= mn) and (x['max'] <= mx), axis=1
        )
    return df_type

def get_type_vars(var_list):
    """
    Processes a list of variables and generates a DataFrame summarizing their properties, such as 
    the number of unique values, count, and range (min, max). Also checks whether numeric variables 
    fit within certain data type ranges (e.g., float32, int32, int16, int8).

    Args:
        var_list (list of tuples): A list of tuples where each tuple contains the following elements:
            - src (str): The source or origin of the variable.
            - s (pd.Series): The pandas Series representing the variable.
            - desc (str): A description of the variable (unused in the function).

    Returns:
        pd.DataFrame: A pandas DataFrame indexed by variable name with the following columns:
            - 'src' (str): The source of the variable.
            - 'dtype' (str): The data type of the variable.
            - 'na' (int): The number of missing values in the variable.
            - 'n_unique' (int): The number of unique values in the variable.
            - 'count' (int): The total count of non-missing values.
            - 'min' (varied): The minimum value of the variable (if applicable).
            - 'max' (varied): The maximum value of the variable (if applicable).
            - 'f32' (bool): Whether the variable can fit within a float32 data type range.
            - 'i32' (bool): Whether the variable can fit within an int32 data type range.
            - 'i16' (bool): Whether the variable can fit within an int16 data type range.
            - 'i8' (bool): Whether the variable can fit within an int8 data type range.

    Example:
        >>> var_list = [
        ...     ('source1', pd.Series([1, 2, 3], dtype='int32'), 'description1', 'Numeric'),
        ...     ('source2', pd.Series(['a', 'b', 'a'], dtype='category'), 'description2', 'Categorical')
        ... ]
        >>> get_type_vars(var_list)
        # Returns a DataFrame with statistics on the variables, including dtype compatibility.
    """
    s_list = list()
    for src, s, desc in var_list:
        if str(s.dtype) == 'category':
            t = 'Categorical'
            if s.dtype.ordered:
                s_list.append(
                    pd.Series(
                        [src, s.name, desc, t, s.isna().sum()] + s.agg(['nunique', 'count', 'min', 'max']).tolist(),
                        index=['src', 'var', 'Description', 'dtype', 'na', 'n_unique', 'count', 'min', 'max']
                    )
                )
            else:
                s_list.append(
                    pd.Series(
                        [src, s.name, desc, t, s.isna().sum()] + s.agg(['nunique', 'count']).tolist(),
                        index=['src', 'var', 'Description', 'dtype', 'na', 'n_unique', 'count']
                    )
                )
        else:
            t = str(s.dtype)
            if t == 'str':
                t = 'String'
            if t == 'datetime64[ns]':
                t = 'Datetime'
            else:
                t = t[:1].upper() + t[1:]
            s_list.append(
                pd.Series(
                    [src, s.name, desc, s.isna().sum(), t] + s.agg(['nunique', 'count', 'min', 'max']).tolist(),
                    index=['src', 'var', 'Description', 'na', 'dtype', 'n_unique', 'count', 'min', 'max']
                )
            )
    df = pd.DataFrame(s_list).set_index('var')
    if 'min' in df.columns:
        for i, mn, mx in [
            ('f32', np.finfo(np.float32).min, np.finfo(np.float32).max),
            ('i32', np.iinfo(np.int32).min, np.iinfo(np.int32).max),
            ('i16', np.iinfo(np.int16).min, np.iinfo(np.int16).max),
            ('i8', np.iinfo(np.int8).min, np.iinfo(np.int8).max)
        ]:
            df[i] = df.loc[~df['dtype'].isin(['String', 'Categorical', 'Datetime'])].apply(
                lambda x: (x['min'] >= mn) and (x['max'] <= mx), axis=1
            )
    return df

def merge_type_df(dfs):
    """
    merge type data frame
    Parameters:
        dfs: list
            merge the type information of DataFrames
    Returns:
        df - return DataFrame
    Examples:
        >>> merge_type_df([pl.scan_csv(i).pipe(get_type_df) for i in glob(*.csv)])
    """
    return pd.concat(dfs, axis=0).pipe(lambda x: x.groupby(x.index).agg(
        {'min': 'min', 'max': 'max', 'na': 'sum', 'count': 'sum', 'n_unique': 'mean', 'dtype': 'max', 'f32': 'all', 'i32': 'all', 'i16': 'all', 'i8': 'all'})
    )

def get_type_pl(df_type, predefine={}, f32=True, i64=False, cat_max=np.inf, txt_cols = []):
    """
    get datatype for Polars DataFrame
    Parameters:
        df_type: pd.DataFrame
            the DataFrame whichi contains type information
        predefine: dict
            the predefined data type
        f32: Boolean
            Use f32 if possible
        i64: Boolean
            use i64 for all integer types
        cat_max: Int
            maximum number of categories for categorical types
        txt_cols: list
            Text type columns
    Returns:
        dict
            the dictionary of type mapping for pl.DataFrame loading
    Examples:
        >>> df_type = pl.scan_csv('test.csv').pipe(get_type_df)
        >>> dt = get_type_pl(df_type, {'id': pl.Int64})
    """
    ret_type = predefine.copy()
    
    for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Date')].index:
        ret_type = pl.Datetime
    if f32:
        for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Int') & (df_type['na'] > 0) & (df_type['f32'])].index:
            ret_type[i] = pl.Float32
        for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Float') & (df_type['f32'])].index:
            ret_type[i] = pl.Float32
    for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Int') & (df_type['na'] > 0)].index:
        ret_type[i] = pl.Float64
    for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Float')].index:
        ret_type[i] = pl.Float64
    if not i64:
        for col, ty in [('i8', pl.Int8), ('i16', pl.Int16), ('i32', pl.Int32)]:
            for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Int') & df_type[col]].index:
                ret_type[i] = ty
        for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Float') & df_type[col]].index:
            ret_type[i] = ty
    for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Int')].index:
        ret_type[i] = pl.Int64
    for i in df_type.loc[~df_type.index.isin(ret_type) & (df_type['n_unique'] <= cat_max) & (~df_type.index.isin(txt_cols))].index:
        ret_type[i] = pl.Categorical
    for i in df_type.loc[~df_type.index.isin(ret_type)].index:
        ret_type[i] = pl.String
    return ret_type

def get_type_pd(df_type, predefine={}, f32=True, i64=False, cat_max=np.inf, txt_cols = []):
    """
    get data type for pandas DataFrame
    Parameters:
        df_type: pd.DataFrame
            the DataFrame whichi contains type information
        predefine: dict
            the predefined data type
        f32: Boolean
            Use f32 if possible
        i64: Boolean
            use i64 for all integer types
        cat_max: Int
            maximum category number for categorical types
        txt_cols: list
            Text type columns
    Examples:
        >>> df_type = pl.scan_csv('test.csv').pipe(get_type_df)
        >>> dt = get_type_pd(df_type, {'id': 'int64'})
    """
    ret_type = predefine.copy()
    
    for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Date')].index:
        ret_type = 'datetime'
    if f32:
        for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Int') & (df_type['na'] > 0) & (df_type['f32'])].index:
            ret_type[i] = 'float32'
        for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Float') & (df_type['f32'])].index:
            ret_type[i] = 'float32'
    for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Int') & (df_type['na'] > 0)].index:
        ret_type[i] = 'float32'
    for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Float')].index:
        ret_type[i] = 'float32'
    if not i64:
        for col, ty in [('i8', 'int8'), ('i16', 'int16'), ('i32', 'int32')]:
            for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Int') & df_type[col]].index:
                ret_type[i] = ty
        for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Float') & df_type[col]].index:
            ret_type[i] = ty
    for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Int')].index:
        ret_type[i] = 'int64'
    for i in df_type.loc[~df_type.index.isin(ret_type) & (df_type['n_unique'] <= cat_max) & (~df_type.index.isin(txt_cols))].index:
        ret_type[i] = 'category'
    for i in df_type.loc[~df_type.index.isin(ret_type)].index:
        ret_type[i] = 'string'
    return ret_type

def apply_with_columns(dfl, proc_list, src=None):
    """
    pl.with_column processing
    Parameters:
        dfl: pl.DataFrame
            Data DataFrame to process
        proc_list: list
            (variable name, pl.Expr, Description)
        df_feat: pd.DataFrame
            Feature DataFrame, if None, does not make feature information
    Returns:
        pl.DataFrame, pd.Dataframe
            Data DataFrame, Feature DataFrame
    Examples:
        >>> target_assign = [
        >>>    ('target', (pl.col('Rings') + 1).log().cast(pl.Float32), "RMSLE 지표를 최적화하기 위한 Rings의 log1p 변환을 하여 target을 만듭니다."),
        >>> ]
        >>> dfl_train, df_feature = with_column_opr(dfl_train, target_assign, 'targetproc1')
        >>> dfl_org, _ = with_column_opr(dfl_org, target_assign)
    """
    df_proc = pd.DataFrame(proc_list, columns=['val', 'proc', 'Description']).set_index('val')
    dfl = dfl.with_columns(**df_proc['proc'])
    if src is not None:
        df_var = pd.concat([
            df_proc[['Description']].assign(src=src), 
            get_type_df(dfl[df_proc.index.tolist()])
        ], axis=1)
        return dfl, df_var
    return dfl

def apply_select(dfl, proc_list, src=None):
    """
    apply select_proc
    Parameters:
        dfl: pl.DataFrame
            Data DataFrame to process
        proc_list: list
            select function list
        desc: Function
            The function provide columns information
        src: str
            The name of source
        df_feat: pd.DataFrame
            Feature DataFrame, if None, does not make feature information
    Returns:
        pl.DataFrame, pd.Dataframe
            Data DataFrame, Feature DataFrame
    Examples:
        >>> dfl_merge = dfl_merge.sort('pca0')
        >>> sig = 1.96
        >>> clip_target = lambda x: x.select(
        >>>             pl.col('target'),
        >>>         ).with_columns(
        >>>             pl.col('target').rolling_mean(101, center=True, min_periods=50).alias('mean_'),
        >>>             pl.col('target').rolling_std(101, center=True, min_periods=50).alias('std_')
        >>>         ).select(
        >>>             pl.col('target').clip(
        >>>                 pl.col('mean_') - pl.col('std_') * sig, 
        >>>                 pl.col('mean_') + pl.col('std_') * sig
        >>>             ).cast(pl.Float32).alias('target_b')
        >>>         )
        >>> proc_list = [(clip_target, 'target의 범위를 pca0를 기준으로 rolling 통계를 이용하여 고정시킵니다.')]
        >>> dfl_merge, df_var = select_opr(dfl_merge, proc_list, 'clip_rolling')
    """
    dfl_list, var_list = list(), list()
    for proc, desc in proc_list:
        dfl_list.append(proc(dfl))
        if src is not None:
            var_list.append(
                pd.DataFrame({
                    'val': dfl_list[-1].columns,
                    'Description': [i[1] for i in desc] if type(desc) == list else [desc],
                }).set_index('val').join(
                    get_type_df(dfl_list[-1])
                )
            )
    dfl_proc = pl.concat(dfl_list, how = 'horizontal')
    del dfl_list
    d = []
    for i in dfl_proc.columns:
        if i in dfl.columns:
            d.append(dfl_proc.drop_in_place(i))
    dfl = dfl.with_columns(*d)
    if src is None:
        return dfl.hstack(dfl_proc)
    else:
        return dfl.hstack(dfl_proc), pd.concat(var_list).assign(src=src)

def apply_pd(df, proc_list, src=None):
    """
    Applies a list of processing functions to a DataFrame and returns the result.

    Parameters:
        df (pd.DataFrame): The input DataFrame to process.
        proc_list (list): A list of tuples, where each tuple contains a processing function and a description.
        src (str, optional): An optional source name to associate with each processed variable. Defaults to None.

    Returns:
        tuple: If `src` is provided, returns a tuple of the concatenated processed DataFrame and type variables.
               Otherwise, returns only the concatenated processed DataFrame.
    """
    type_list, var_list = list(), list()
    for proc, desc in proc_list:
        var_list.append(proc(df))
        if src is not None:
            if type(var_list[-1]) == pd.Series:
                type_list.append([src, var_list[-1], desc])
            else:
                for i in var_list[-1].columns:
                    type_list.append([src, var_list[-1][i], desc.get(i, '')])
    if src is not None:
        return pd.concat(var_list, axis=1), get_type_vars(type_list)
    return pd.concat(var_list, axis=1)

def apply_pd_procs(df, s_procs):
    """
    Applies a series of processing steps to a DataFrame and returns the updated DataFrame 
    along with any unprocessed steps.

    This function takes a dictionary of processing steps, where each key is a step name 
    and each value is a list of tuples containing processing functions and their descriptions.
    It iteratively applies these processing steps to the DataFrame, retrying failed steps 
    until no more progress can be made.

    Parameters:
        df (pd.DataFrame): The input DataFrame to process.
        s_procs (dict): A dictionary where keys are step names and values are lists of tuples. 
            Each tuple consists of:
                - A processing function (callable): The function to apply to the DataFrame.
                - A description (any): A description associated with the processing function.

    Returns:
        tuple:
            - pd.DataFrame: The DataFrame after all successfully processed steps.
            - list: A list of unprocessed steps that could not be applied, where each step is 
              represented as a tuple of step name and corresponding processing list.
    """
    nq = [(k, v) for k, v in s_procs.items()]
    while len(nq) > 0:
        steps = list()
        success = list()
        q = nq.copy()
        nq = list()
        proccessed = list()
        while (len(q) > 0):
            p = q.pop()
            try:
                proccessed.append(apply_pd(df, p[1]))
                success.append(p)
            except Exception as e:
                nq.append(p)
        if len(proccessed) > 0:
            df = join_and_assign(df, pd.concat(proccessed, axis=1))
            steps.append(success)
        else:
            break
    return df, nq
    
def join_and_assign(df1, df2):
    """
    Joins columns from the first DataFrame to the second DataFrame if they do not already exist in the second DataFrame.

    Args:
        df1 (pd.DataFrame): The source DataFrame containing columns to merge.
        df2 (pd.DataFrame): The target DataFrame to which columns from `df1` will be joined if not present.

    Returns:
        pd.DataFrame: The resulting DataFrame with `df1`'s columns added to `df2` where they were missing.
    """
    to_merge = [i for i in df1.columns if i not in df2.columns]
    if len(to_merge) == 0: return df2.copy()
    return df1[to_merge].join(df2)

class PD_Vars():
    """
    A class for managing and applying processing functions to DataFrames, along with metadata tracking.

    Attributes:
        file_name (str): The name of the file to save or load processing data.
        df_var (pd.DataFrame): A DataFrame containing metadata about variables and their sources.
        d_procs (OrderedDict): A dictionary storing processing steps with their associated names.
        modified (bool): A flag indicating if the processing data has been modified.
    """
    
    def __init__(self, file_name, df_var):
        """
        Initializes the PD_Vars object.

        Parameters:
            file_name (str): The name of the file for saving and loading processing data.
            df_var (pd.DataFrame): A DataFrame containing variable metadata.

        """
        self.file_name = file_name
        if df_var is not None:
            self.df_var = df_var.copy()
        self.d_procs = OrderedDict()
        self.modified = False

    def put_proc(self, name, df, proc_list, replace=False):
        """
        Adds or updates a processing step.

        Parameters:
            name (str): The name of the processing step.
            df (pd.DataFrame): The DataFrame to which the processing step is applied.
            proc_list (list): A list of tuples containing processing functions and their descriptions.
                (function, description), if function returns multiple variables, the type description is dict of which key is variable name and value is the description.
            replace (bool, optional): If True, replaces the existing processing step. Defaults to False.

        Returns:
            pd.DataFrame: The updated DataFrame after applying the processing step.
        """
        if name in self.d_procs and not replace:
            src_vars = self.df_var[self.df_var['src'] == name].index
            if src_vars.isin(df.columns).all():
                return df
            return join_and_assign(
                df, apply_pd(df, self.d_procs[name])
            )
        self.modified = True
        df_ret, df_new_var = apply_pd(df, proc_list, name)
        self.d_procs[name] = proc_list
        self.df_var = pd.concat([
            self.df_var.drop(columns=[i for i in self.df_var.columns if i not in df_new_var.columns]), df_new_var
        ], axis = 0).groupby(level=0).last()
        return join_and_assign(
            df, df_ret
        )
    
    def proc(self, name, df):
        """
        Applies a named processing step to a DataFrame.

        Parameters:
            name (str): The name of the processing step.
            df (pd.DataFrame): The DataFrame to process.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        return apply_pd(df, self.d_procs[name])
        
    def procs_all(self, df):
        """
        Applies all stored processing steps to a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to process.

        Returns:
            tuple:
                - pd.DataFrame: The DataFrame after applying all processing steps.
                - list: A list of unprocessed steps that could not be applied.
        """
        return apply_pd_procs(df, self.d_procs)

    def del_proc(self, name):
        """
        Deletes a named processing step and its associated metadata.

        Args:
            name (str): The name of the processing step to delete.
        """
        if name not in self.d_procs:
            return
        del self.d_procs[name]
        to_del = self.df_var.loc[self.df_var['src'] == name].index
        self.df_var.drop(index=to_del, inplace=True)

    def reorder(self, names):
        """
        Reorders the processing steps based on a given sequence.

        Args:
            names (list): A list of processing step names in the desired order.
        """
        for i in names:
            if i in self.d_procs:
                self.d_procs.move_to_end(i)

    def clear(self, df):
        """
        Removes columns from the DataFrame that are not tracked in the metadata.

        Args:
            df (pd.DataFrame): The DataFrame to clear.

        Returns:
            pd.DataFrame: The cleaned DataFrame with only tracked columns.
        """
        d = [i for i in df.columns if i not in self.df_var.index]
        if len(d) == 0:
            return df
        else:
            return df.drop(columns = d)

    def clear_procs(self, df):
        d = [i for i in self.df_var.index if i not in df.columns]
        if len(d) != 0:
            self.df_var.drop(index = d, inplace=True)
            
    def save(self, overwrite=False):
        """
        Saves the processing steps and metadata to a file.

        Args:
            overwrite (bool, optional): If True, saves even if no modifications were made. Defaults to False.
        """
        if not self.modified and not overwrite:
            return
        with open(self.file_name + '.dill', 'wb') as f:
            dill.dump({
                'var': self.df_var,
                'd_procs': self.d_procs,
            }, f)
        self.modified = False

    @staticmethod
    def load(file_name):
        """
        Loads processing steps and metadata from a file.

        Args:
            file_name (str): The name of the file to load.

        Returns:
            PD_Vars: The loaded PD_Vars object.
        """
        with open(file_name + '.dill', 'rb') as f:
            d = dill.load(f)
        ret = PD_Vars(file_name, d['var'])
        ret.d_procs = d['d_procs']
        return ret

    @staticmethod
    def load_or_create(file_name, df_var):
        """
        Loads a PD_Vars object from a file or creates a new one if the file does not exist.

        Args:
            file_name (str): The name of the file to load or create.
            df_var (pd.DataFrame): A DataFrame containing variable metadata.

        Returns:
            PD_Vars: The loaded or newly created PD_Vars object.
        """
        if os.path.exists(file_name + '.dill'):
            return PD_Vars.load(file_name)
        return PD_Vars(file_name, df_var)

def combine_cat(df, delimiter=''):
    """
    Combines multiple categorical columns in a DataFrame into a single categorical variable, in efficient way. 

    Parameters:
        df (pd.DataFrame): DataFrame where each column is of categorical dtype.
        delimiter (str): Delimiter

    Returns:
        pd.Series: A Series containing a new categorical variable that represents 
                   the unique combination of all input categorical columns.
    """
    return pd.Series(
        pd.Categorical.from_codes(
            df.apply(lambda x: x.cat.codes).dot(df.nunique().shift(1).fillna(1).astype('int').cumprod()), 
            [delimiter.join(i[::-1]) for i in product(*df[df.columns[::-1]].apply(lambda x: x.cat.categories.astype('str').tolist(), result_type='reduce'))]
        ), index=df.index
    )

def replace_cat(s, rule):
    """
    Replaces the categories in a pandas Categorical Series based on a given rule.

    Args:
        s (pd.Series): A pandas Series with categorical dtype (i.e., `pd.Categorical`).
        rule (Union[Dict[str, str], Callable[[str], str]]): A mapping or function that defines the 
            new category replacements.
            - If `rule` is a dictionary, the keys are the original categories and the values are the 
              replacements.
            - If `rule` is a function, it takes an original category and returns the new category.

    Returns:
        pd.Series: A new pandas Categorical object with the categories replaced 
        according to the given rule.

    Example:
        >>> s = pd.Series(['a', 'b', 'c'], dtype='category')
        >>> rule = {'a': 'x', 'b': 'x'}
        >>> replace_cat(s, rule)
        [x, x, c]
        Categories (2, object): [x, c]

        Or using a function:

        >>> rule = lambda x: x.upper()
        >>> replace_cat(s, rule)
        [A, B, C]
        Categories (3, object): [A, B, C]

    """
    code_replace = {}
    d = {}
    for c, n in zip(
        range(len(s.cat.categories)), 
        s.cat.categories
    ):
        new_cat = rule.get(n, n) if type(rule) == dict else rule(n)
        if new_cat in code_replace:
            d[c] = code_replace[new_cat]
        else:
            d[c] = len(code_replace)
            code_replace[new_cat] = len(code_replace)
    return pd.Series(pd.Categorical.from_codes(
            s.cat.codes.map(d), list(code_replace.keys()), ordered=s.cat.ordered
        ), index=s.index)

def rearrange_cat(s_cat, cat_type, repl_rule, use_set=False):
    """
    Rearranges the categories of a pandas Categorical series based on a provided category type
    and a replacement rule for missing categories.

    Args:
        s_cat (pd.Series): A pandas Series of categorical values that need to be rearranged.
        cat_type (pd.api.types.CategoricalDtype): The target CategoricalDtype defining the desired
            order and structure of categories.
        repl_rule (callable): A function that defines a rule for handling categories in `s_cat`
            that are not found in `cat_type`. The function should take two arguments:
            `cat_vals` (the array of category values from `cat_type`) and `x` (a missing category
            from `s_cat`). It returns the value to replace the missing category with.
        use_set(Boolean): provide cat_vals with set
    Returns:
        pd.Categorical: A new pandas Categorical object with the categories of `s_cat` rearranged
        according to `cat_type`, and with missing categories replaced based on `repl_rule`.

    Example:
        >>> cat_type = pd.api.types.CategoricalDtype(categories=["a", "b", "c"])
        >>> s_cat = pd.Series(pd.Categorical(["a", "d", "b"]))
        >>> def repl_rule(cat_vals, x):
        ...     return 0  # Default to the first category if not found
        >>> rearrange_cat(s_cat, cat_type, repl_rule)
        [a, a, b]
        Categories (3, object): [a, b, c]
    """
    cat_vals = cat_type.categories.values
    s_map = pd.Series(np.arange(len(cat_vals)), cat_vals)
    if use_set:
        cat_vals_s = set(cat_vals)
        s_cat_map = pd.Series(s_cat.cat.categories.values, s_cat.cat.categories.values).apply(
            lambda x: s_map[x] if x in s_map else repl_rule(cat_vals_s, x)
        )
    else:
        s_cat_map = pd.Series(s_cat.cat.categories.values, s_cat.cat.categories.values).apply(
            lambda x: s_map[x] if x in s_map else repl_rule(cat_vals, x)
        )
    notna = s_cat.notna()
    return s_cat.loc[notna].pipe(
        lambda x: pd.Series(pd.Series(pd.Categorical.from_codes(x.map(s_cat_map), cat_vals), index=x.index), index=s_cat.index)
    ) if notna.sum() != len(s_cat) else pd.Series(pd.Categorical.from_codes(s_cat.map(s_cat_map), cat_vals), index=s_cat.index)

def split_preprocess_var(s_names, org_names):
    return s_names.str.split('__|_').apply(
        lambda x: (x, ([i for i in range(len(x), 0, -1) if '_'.join(x[1:i]) in org_names] + [-1])[0])
    ).apply(
        lambda x: pd.Series(['_'.join(x[0][:x[1]]), ''.join(x[0][x[1]:])], index=['var1', 'var2'])
    )