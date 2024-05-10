import pandas as pd
import numpy as np
import polars as pl
import polars.selectors as cs
from functools import partial

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
    df_type = df_type.pivot(index='feature', columns='stat', values='value')\
                .to_pandas().set_index('feature').join(
                    pd.Series([str(i) for i in df.dtypes], index=df.columns, name='dtype')
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
            maximum category number for categorical types
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

def with_columns_opr(dfl, proc_list, df_feat=None):
    """
    pl.with_column processing
    Parameters:
        dfl: pl.DataFrame
            Data DataFrame to process
        proc_list: list
            (src, variable name, pl.Expr, Description)
        df_feat: pd.DataFrame
            Feature DataFrame, if None, does not make feature information
    Returns:
        pl.DataFrame, pd.Dataframe
            Data DataFrame, Feature DataFrame
    Examples:
        >>> target_assign = [
        >>>    ('targetproc1', 'target', (pl.col('Rings') + 1).log().cast(pl.Float32), "RMSLE 지표를 최적화하기 위한 Rings의 log1p 변환을 하여 target을 만듭니다."),
        >>> ]
        >>> dfl_train, df_feature = with_column_opr(dfl_train, target_assign, df_feature)
        >>> dfl_org, _ = with_column_opr(dfl_org, target_assign)
    """
    df_proc = pd.DataFrame(proc_list, columns=['src', 'val', 'proc', 'Description']).set_index('val')
    dfl = dfl.with_columns(**df_proc['proc'])
    if df_feat is not None:
        df_feat = pd.concat([
            df_proc[['src', 'Description']], 
            pd.Series([str(i) for i in dfl[df_proc.index.tolist()].dtypes], index=df_proc.index, name='type')
        ], axis=1).pipe(
            lambda x: pd.concat([df_feat, x], axis=0)
        )
    return dfl, df_feat

def apply_processor(dfl, processor, X_val, info_prov, df_feat=None):
    """
    sklearn.preprocessing processing
    Parameters:
        dfl: pl.DataFrame
            Data DataFrame to process
        processor: object
            Sklearn Preprocessor object
        X_val: list
            Proprocessign target variable names
        info_prov: Function
            The function provide columns information
        df_feat: pd.DataFrame
            Feature DataFrame, if None, does not make feature information
    Returns:
        pl.DataFrame, pd.Dataframe
            Data DataFrame, Feature DataFrame
    Examples:
        >>> X_std = df_feature.query('src == "origin" and type == "Float32"').index.to_series().replace({'Height': 'Height_n'}).tolist()
        >>> pipe_std_pca = make_pipeline(
        >>>     StandardScaler(), 
        >>>     ColumnTransformer([
        >>>         ('std', 'passthrough', np.arange(len(X_std)).tolist()), 
        >>>         ('pca', PCA(n_components=4), np.arange(len(X_std)).tolist())
        >>>     ])
        >>> )
        >>> def info_prov(p, v):
        >>>     if p == 'pca':
        >>>         return ('pca', v, 'Size features PCA component ' + v, pl.Float32)
        >>>     return ('std', v, 'StandardScaler: ' + v, pl.Float32)
        >>> dfl = apply_processor(dfl, processor=pipe_std_pca, X_val=X_std, info_prov=info_prov, df_feat = df_feature)
    """
    entity = list()
    if df_feat is not None:
        processor.fit(dfl[X_val])
    for i in processor.get_feature_names_out():
        sp = i.split('__')
        if len(sp) > 1: 
            p, v = sp[0], sp[1]
        else:
            p, v = sp[0], ''
        entity.append(info_prov(p, v))
    df_feat_ = pd.DataFrame(entity, columns=['src', 'val', 'Description', 'dt']).set_index('val')
    dfl_proc = pl.DataFrame(
        processor.transform(dfl.select(cs.by_name(X_val))),
        schema = df_feat_['dt'].to_dict()
    )
    if df_feat is not None:
        df_feat_ = df_feat_.drop(columns=['dt']).join(pd.Series([str(i) for i in dfl_proc.dtypes], index=df_feat_.index.tolist(), name='type'))
        df_feat = pd.concat([df_feat, df_feat_], axis=0)
    d = []
    for i in dfl_proc.columns:
        if i in dfl.columns:
            d.append(dfl_proc.drop_in_place(i))
    dfl = dfl.with_columns(*d)
    return dfl.hstack(dfl_proc), df_feat

def select_opr(dfl, select_proc, desc, df_feat=None):
    """
    apply select_proc
    Parameters:
        dfl: pl.DataFrame
            Data DataFrame to process
        processor: Function
            dfl proccesing function
        X_val: list
            Proprocessign target variable names
        info_prov: Function
            The function provide columns information
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
        >>> desc = [('clip_rolling', 'target의 범위를 pca0를 기준으로 rolling 통계를 이용하여 고정시킵니다.')]
        >>> dfl_merge, df_feature = select_opr(dfl_merge, clip_target, desc, df_feature)
    """
    dfl_proc = select_proc(dfl)
    if df_feat is not None:
        df_feat_ = pd.DataFrame({
            'val': dfl_proc.columns,
            'type': [str(i) for i in dfl_proc.dtypes], 
            'Description': [i[1] for i in desc],
            'src': [i[0] for i in desc],
        }).set_index('val')
        df_feat = pd.concat([df_feat, df_feat_], axis=0)
    d = []
    for i in dfl_proc.columns:
        if i in dfl.columns:
            d.append(dfl_proc.drop_in_place(i))
    dfl = dfl.with_columns(*d)
    return dfl.hstack(dfl_proc), df_feat

def apply_procs(dfl, procs, df_feat=None):
    """
    apply preprocessors
    Parameters:
        dfl: pl.DataFrame
            Data DataFrame to process
        procs: list
            list of processor
        df_feat: pd.DataFrame
            Feature DataFrame, if None, does not fit model and make feature information
    Returns:
        pl.DataFrame, pd.Dataframe
            Data DataFrame, Feature DataFrame
    Examples:
        >>> procs = list()
        >>> feat_assign = [
        >>>     ('preproc1', 'Height_n', pl.col('Height').clip(0.004, 0.35), "Clip Height as Height_n")
        >>> ]
        >>> procs.append(partial(with_column_opr, proc_list=feat_assign))
        >>>
        >>> X_std = df_feature.query('src == "origin" and type == "Float32"').index.to_series().replace({'Height': 'Height_n'}).tolist()
        >>> pipe_std_pca = make_pipeline(
        >>>     StandardScaler(), 
        >>>     ColumnTransformer([
        >>>         ('std', 'passthrough', np.arange(len(X_std)).tolist()), 
        >>>         ('pca', PCA(n_components=4), np.arange(len(X_std)).tolist())
        >>>     ])
        >>> )
        >>> def info_prov(p, v):
        >>>     if p == 'pca':
        >>>         return ('pca', v, 'Size features PCA component ' + v, pl.Float32)
        >>>     return ('std', v, 'StandardScaler: ' + v, pl.Float32)
        >>> procs.append(partial(apply_processor, processor=pipe_std_pca, X_val=X_std, info_prov=info_prov))
        >>>
        >>> X_ord = df_feature.query('src == "origin" and type == "Categorical"').index.to_list()
        >>> ord_enc = OrdinalEncoder(dtype=np.int32, handle_unknown='use_encoded_value', unknown_value=-1)
        >>> procs.append(partial(apply_processor, processor=ord_enc, X_val=X_ord, info_prov=ord_prov))
        >>> dfl_train, df_feature = apply_procs(dfl_train, procs, df_feature)
        >>> dfl_org, _ = apply_procs(dfl_org, procs)
    """
    if df_feat is None:
        for proc in procs:
            dfl, _ = dfl.pipe(proc)
    else:
        for proc in procs:
            dfl, df_feat = dfl.pipe(partial(proc, df_feat=df_feat))
    return dfl, df_feat

def ord_prov(p, v):
    """
    Information provider for Oridinal Encoder
    """
    return ('ord', p, 'OrdinalEncoder: ' + p, pl.Int16)

def ohe_prov(p, v):
    """
    Information provider for OneHot Encoder
    """
    return ('ohe', p, 'OneHotEncoder: ' + v, pl.Int8)
