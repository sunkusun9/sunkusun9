from ._data_wrapper import unwrap

def resolve_columns(data, X, y=None, org_X = None):
    """X와 y를 실제 컬럼 리스트로 변환"""
    import re
    columns = data.get_columns()

    # y 처리 (y가 있으면 X에서 제외할 컬럼)
    y_cols = []
    if y is not None:
        if isinstance(y, slice):
            y_cols = columns[y]
        elif isinstance(y, list):
            y_cols = y
        else:
            y_cols = [y]

    # X 처리
    if X is None:
        # y가 있으면 y를 제외한 모든 컬럼
        if y is not None:
            return [col for col in columns if col not in y_cols]
        else:
            return columns
    elif isinstance(X, re.Pattern):
        # 정규 표현식 패턴이면 매칭되는 컬럼만 선택
        return [col for col in columns if X.match(col)]
    elif callable(X):
        # 함수면 columns를 전달하고 Boolean array를 받아서 True인 컬럼만 선택
        if org_X is None:
            mask = X(columns)
        else:
            mask = X(columns, org_X=org_X)
        return [col for col, keep in zip(columns, mask) if keep]
    elif isinstance(X, slice):
        # slice 객체면 컬럼을 슬라이싱
        return columns[X]
    elif isinstance(X, list):
        # 리스트면 그대로 반환
        return X
    else:
        # 단일 값이면 리스트로 변환
        return [X]

class TransformProcessor():
    def __init__(self, node, transformer, X = None, y = None, adapter = None, **args):
        self.node = node
        self.transformer = transformer
        self.params = args
        self.X = X
        self.y = y
        self.adapter = adapter
        self.output_vars = None

    def fit(self, train, valid):
        self.X_ = resolve_columns(train, self.X, self.y)
        self.obj = self.transformer(**self.params)
        fit_params = {}

        # DataWrapper에서 컬럼 선택
        train_X = unwrap(train.select_columns(self.X_))

        if self.y is None:
            if self.adapter is not None:
                valid_X = unwrap(valid.select_columns(self.X_)) if valid is not None else None
                if valid is None:
                    fit_params = self.adapter.get_fit_params(X_train = train_X)
                else:
                    fit_params = self.adapter.get_fit_params(X_train = train_X, X_eval = valid_X)
            self.obj.fit(train_X, **fit_params)
        else:
            train_y = unwrap(train.select_columns(self.y))
            if self.adapter is not None:
                valid_X = unwrap(valid.select_columns(self.X_)) if valid is not None else None
                valid_y = unwrap(valid.select_columns(self.y)) if valid is not None else None
                if valid is None:
                    fit_params = self.adapter.get_fit_params(X_train = train_X, y_train = train_y)
                else:
                    fit_params = self.adapter.get_fit_params(X_train = train_X, y_train = train_y, X_eval = valid_X, y_eval = valid_y)
            self.obj.fit(train_X, train_y, **fit_params)
        # 컬럼명 결정 (get_feature_names_out이 있으면 사용)
        if hasattr(self.obj, 'get_feature_names_out'):
            column_names = self.obj.get_feature_names_out().tolist()
            column_names = [f"{self.node.name}__{col}" for col in column_names]
        else:
            column_names = None

        if column_names is not None:
            self.output_vars = column_names
        return self

    def fit_process(self, train, valid):
        self.X_ = resolve_columns(train, self.X, self.y)
        self.obj = self.transformer(**self.params)
        fit_params = {}

        # DataWrapper에서 native로 변환
        train_X = unwrap(train.select_columns(self.X_))
        train_index = train.get_index()

        if self.y is None:
            if self.adapter is not None:
                valid_X = unwrap(valid.select_columns(self.X_)) if valid is not None else None
                if valid is None:
                    fit_params = self.adapter.get_fit_params(X_train = train_X)
                else:
                    fit_params = self.adapter.get_fit_params(X_train = train_X, X_eval = valid_X)
            result = self.obj.fit_transform(train_X, **fit_params)
        else:
            train_y = unwrap(train.select_columns(self.y))
            if self.adapter is not None:
                valid_X = unwrap(valid.select_columns(self.X_)) if valid is not None else None
                valid_y = unwrap(valid.select_columns(self.y)) if valid is not None else None
                if valid is None:
                    fit_params = self.adapter.get_fit_params(X_train = train_X, y_train = train_y)
                else:
                    fit_params = self.adapter.get_fit_params(X_train = train_X, y_train = train_y, X_eval = valid_X, y_eval = valid_y)
            result = self.obj.fit_transform(train_X, train_y, **fit_params)

        # train의 Wrapper 타입으로 변환
        train_wrapper_class = type(train)
        return train_wrapper_class.from_output(result, self.output_vars, train_index)

    def process(self, data):
        # DataWrapper에서 native로 변환
        data_X = unwrap(data.select_columns(self.X_))
        data_index = data.get_index()

        if self.y is None:
            result = self.obj.transform(data_X)
        else:
            data_y = unwrap(data.select_columns(self.y))
            result = self.obj.transform(data_X, data_y)

        # data의 Wrapper 타입으로 변환
        data_wrapper_class = type(data)
        return data_wrapper_class.from_output(result, self.output_vars, data_index)

class PredictProcessor():
    def __init__(self, node, estimator, X=None, y=None, method='predict', adapter = None, **args):
        self.node = node
        self.estimator = estimator
        self.params = args
        self.X = X
        self.y = y
        self.method = method
        self.output_vars = None
        self.adapter = adapter

    def fit(self, train, valid):
        self.X_ = resolve_columns(train, self.X, self.y)
        self.obj = self.estimator(**self.params)
        fit_params = {}

        # DataWrapper에서 컬럼 선택
        train_X = unwrap(train.select_columns(self.X_))

        if self.y is None:
            if self.adapter is not None:
                valid_X = unwrap(valid.select_columns(self.X_)) if valid is not None else None
                if valid is None:
                    fit_params = self.adapter.get_fit_params(X_train = train_X)
                else:
                    fit_params = self.adapter.get_fit_params(X_train = train_X, X_eval = valid_X)
            # 비지도학습 with specific columns
            self.obj.fit(train_X, **fit_params)
        else:
            train_y = unwrap(train.select_columns(self.y))
            if self.adapter is not None:
                valid_X = unwrap(valid.select_columns(self.X_)) if valid is not None else None
                valid_y = unwrap(valid.select_columns(self.y)) if valid is not None else None
                if valid is None:
                    fit_params = self.adapter.get_fit_params(X_train = train_X, y_train = train_y)
                else:
                    fit_params = self.adapter.get_fit_params(X_train = train_X, y_train = train_y, X_eval = valid_X, y_eval = valid_y)
            # 지도학습
            self.obj.fit(train_X, train_y, **fit_params)

        if self.method == 'predict':
            # y 변수명 결정
            if self.y is None:
                y_name = 'prediction'
            elif isinstance(self.y, list):
                y_name = '_'.join(self.y)
            else:
                y_name = self.y

            col_name = f"{self.node.name}__{y_name}"
            self.output_vars = [col_name]
        elif self.method == 'predict_proba':
            # y 변수명 결정
            if self.y is None:
                y_name = 'prediction'
            elif isinstance(self.y, list):
                y_name = '_'.join(self.y)
            else:
                y_name = self.y

            columns = [f"{self.node.name}__{y_name}_{i}" for i in self.obj.classes_]
            self.output_vars = columns
        return self

    def fit_process(self, train, valid):
        self.X_ = resolve_columns(train, self.X, self.y)
        self.obj = self.estimator(**self.params)
        fit_params = {}

        # DataWrapper에서 native로 변환
        train_X = unwrap(train.select_columns(self.X_))
        train_index = train.get_index()

        if self.y is None:
            if self.adapter is not None:
                valid_X = unwrap(valid.select_columns(self.X_)) if valid is not None else None
                if valid is None:
                    fit_params = self.adapter.get_fit_params(X_train = train_X)
                else:
                    fit_params = self.adapter.get_fit_params(X_train = train_X, X_eval = valid_X)
            # 비지도학습 with specific columns
            predictions = self.obj.fit_predict(train_X, **fit_params)
        else:
            # 지도학습
            train_y = unwrap(train.select_columns(self.y))
            if self.adapter is not None:
                valid_X = unwrap(valid.select_columns(self.X_)) if valid is not None else None
                valid_y = unwrap(valid.select_columns(self.y)) if valid is not None else None
                if valid is None:
                    fit_params = self.adapter.get_fit_params(X_train = train_X, y_train = train_y)
                else:
                    fit_params = self.adapter.get_fit_params(X_train = train_X, y_train = train_y, X_eval = valid_X, y_eval = valid_y)
            predictions = self.obj.fit_predict(train_X, train_y, **fit_params)

        # 컬럼명 결정
        if self.y is None:
            y_name = 'prediction'
        elif isinstance(self.y, list):
            y_name = '_'.join(self.y)
        else:
            y_name = self.y

        col_name = f"{self.node.name}__{y_name}"
        column_names = [col_name]
        self.output_vars = column_names

        # train의 Wrapper 타입으로 변환
        train_wrapper_class = type(train)
        return train_wrapper_class.from_output(predictions, column_names, train_index)

    def process(self, data):
        # DataWrapper에서 native로 변환
        data_X = unwrap(data.select_columns(self.X_))
        data_index = data.get_index()

        if self.method == 'predict':
            predictions = self.obj.predict(data_X)
            # 컬럼명은 fit에서 이미 결정됨
            column_names = self.output_vars

        elif self.method == 'predict_proba':
            if not hasattr(self.obj, 'predict_proba'):
                raise Exception(f"Model {self.estimator.__name__} does not support predict_proba")

            predictions = self.obj.predict_proba(data_X)
            # 컬럼명은 fit에서 이미 결정됨
            column_names = self.output_vars

        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'predict' or 'predict_proba'")

        # data의 Wrapper 타입으로 변환
        data_wrapper_class = type(data)
        return data_wrapper_class.from_output(predictions, column_names, data_index)
