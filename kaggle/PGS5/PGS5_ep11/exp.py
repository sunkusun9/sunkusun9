import polars as pl
import pandas as pd
import numpy as np
import pickle as pkl
import adapter
from data_wrapper import DataWrapper, wrap, unwrap
from sklearn.model_selection import ShuffleSplit

def resolve_columns(data, X, y=None, org_X = None):
    """X와 y를 실제 컬럼 리스트로 변환"""
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

class NodeGroup():
    def __init__(self, experimenter, name, processor = None, edges = list(), X = None, y = None, method = 'transform', parent_grp = None, adapter = 'default', params = None):
        self.experimenter = experimenter
        self.name = name
        self.processor = processor
        self.edges = edges if isinstance(edges, list) else [edges]
        self.X = X
        self.y = y
        self.method = method
        self.params = params if params is not None else {}
        self.nodes = []
        self.parent_grp = parent_grp
        self.child_grps = []
        self.adapter = adapter

    def get_attrs(self):
        attrs = {}
        parent_attrs = self.parent_grp.get_attrs() if self.parent_grp is not None else {}
        parent_edges = parent_attrs.get('edges', list())
        if parent_edges is None:
            parent_edges = list()
        attrs['edges'] = parent_edges + self.edges
        # 다른 속성들은 None이 아니면 현재 값, None이면 부모로 올라가면서 찾기
        for attr_name in ['processor', 'X', 'y', 'method']:
            current_value = getattr(self, attr_name, None)
            if current_value is not None:
                attrs[attr_name] = current_value
            elif self.parent_grp is not None:
                attrs[attr_name] = parent_attrs.get(attr_name, None)
            else:
                attrs[attr_name] = None

        # params는 부모의 params를 가져와서 현재 params로 override
        if self.parent_grp is not None:
            parent_params = parent_attrs.get('params', {})
            attrs['params'] = {**parent_params, **self.params}
        else:
            attrs['params'] = self.params

        return attrs

class Node():
    def __init__(
        self, experimenter, name, processor, edges, X = None, y = None, method = 'transform',
        use_cache = True, grp_name = None, org_attr = None, adapter = 'default', params = None
    ):
        self.experimenter = experimenter
        self.name = name
        self.grp_name = grp_name  # 속한 그룹 이름
        self.org_attr = org_attr  # 원본 속성 (processor, edges, X, y, method, params)
        self.processor = processor
        self.method = method
        self.edges = edges
        self.params = params if params is not None else {}
        self.X = X
        self.y = y
        self.use_cache = use_cache
        self.adapter = adapter
        self.build()

    def build(self):
        if self.method in ['transform', 'predict', 'predict_proba']:
            self._fit()
        elif self.method in ['fit_transform', 'fit_predict']:
            self._fit_process()
        else:
            raise ValueError(f"Unknown processor_type: {self.method}")

    def _fit(self):
        self.cache_idx = -1
        self.cache = None
        self.objs_ = list()

        # adapter 인스턴스 가져오기
        if self.adapter == 'default':
            adapter_ = adapter.get_adapter(self.processor)
        else:
            adapter_ = self.adapter

        # 전체 검증 수 계산
        total_folds = sum(len(self.experimenter.train_idx_list[i]) for i in range(len(self.experimenter.train_idx_list)))
        current = 0

        for train in self.experimenter.split(self.edges):
            sub = list()
            for (train_t, train_v), _ in train:
                # 진행상황 출력
                current += 1
                percentage = int(current * 100 / total_folds)
                print(f"\r[{self.name}] Building: {current}/{total_folds} ({percentage}%)", end='', flush=True)
                if self.method in ['transform', 'fit_transform']:
                    obj = TransformProcessor(self, self.processor, X = self.X, y = self.y, adapter = adapter_, **self.params)
                else:
                    obj = PredictProcessor(self, self.processor, X = self.X, y = self.y, method = self.method, adapter = adapter_, **self.params)

                # 수행시간 측정
                import time
                start_time = time.time()
                obj.fit(train_t, train_v)
                elapsed_time = time.time() - start_time

                # 수행 정보 저장
                info = {
                    'fit_time': elapsed_time,
                    'train_shape': train_t.get_shape() if train_t is not None else None,
                    'train_v_shape': train_v.get_shape() if train_v is not None else None
                }
                sub.append((obj, None, info))
            self.objs_.append(sub)

        # 완료 메시지 출력
        print(f"\r[{self.name}] Building: {total_folds}/{total_folds} (100%) ✓ Complete")

    def _fit_process(self):
        self.cache_idx = -1
        self.cache = None
        self.objs_ = list()

        # adapter 인스턴스 가져오기
        if self.adapter == 'default':
            adapter_ = adapter.get_adapter(self.processor)
        else:
            adapter_ = self.adapter

        # 전체 검증 수 계산
        total_folds = sum(len(self.experimenter.train_idx_list[i]) for i in range(len(self.experimenter.train_idx_list)))
        current = 0

        for train in self.experimenter.split(self.edges):
            sub = list()
            for (train_t, train_v), _ in train:
                if self.method in ['transform', 'fit_transform']:
                    obj = TransformProcessor(self, self.processor, X = self.X, y = self.y, adapter = adapter_, **self.params)
                else:
                    obj = PredictProcessor(self, self.processor, X = self.X, y = self.y, method = self.method, adapter = adapter_, **self.params)

                # 수행시간 측정
                import time
                start_time = time.time()
                result = obj.fit_process(train_t, train_v)
                elapsed_time = time.time() - start_time

                info = {
                    'fit_time': elapsed_time,
                    'train_shape': train_t.get_shape() if train_t is not None else None,
                    'train_v_shape': train_v.get_shape() if train_v is not None else None
                }
                sub.append((obj, result, info))

                # 진행상황 출력
                current += 1
                percentage = int(current * 100 / total_folds)
                print(f"\r[{self.name}] Building: {current}/{total_folds} ({percentage}%)", end='', flush=True)
            self.objs_.append(sub)

        # 완료 메시지 출력
        print(f"\r[{self.name}] Building: {total_folds}/{total_folds} (100%) ✓ Complete")

    def get_data(self, idx, v = None):
        if self.cache_idx == idx and self.cache is not None:
            def ret_func():
                for i in self.cache:
                    yield i
            return ret_func()
        it = self.experimenter.get_data(idx, self.edges)
        sub = self.objs_[idx]
        def ret_func():
            if self.use_cache:
                self.cache = list()
            else:
                self.cache = None
            for ((train_t, train_v), valid), (obj, train_, info) in zip(it, sub):
                # train data 처리
                if train_ is None:
                    train_result = obj.process(train_t)
                else:
                    train_result = train_

                # train_v data 처리
                if train_v is not None:
                    train_v_result = obj.process(train_v)
                else:
                    train_v_result = None

                # valid data 처리 (외부 fold의 valid)
                valid_result = obj.process(valid)

                # 필요하면 컬럼 필터링
                if v is not None:
                    X = resolve_columns(train_result, v, org_X = obj.X_)
                    train_result = train_result.select_columns(X)
                    if train_v_result is not None:
                        train_v_result = train_v_result.select_columns(X)
                    valid_result = valid_result.select_columns(X)

                yld = (train_result, train_v_result), valid_result
                if self.cache is not None:
                    self.cache.append(yld)
                yield yld
        if self.use_cache:
            self.cache_idx = idx
        return ret_func()

class RootNode():
    def __init__(self, experimenter, data):
        self.experimenter = experimenter
        self.data = data

    def get_data(self, idx, v = None):
        outer_valid_data = self.data.iloc(self.experimenter.valid_idx_list[idx])

        def ret_func():
            for train_v_idx, valid_v_idx in self.experimenter.train_idx_list[idx]:
                if v is None:
                    train_data = self.data.iloc(train_v_idx)
                    train_v_data = self.data.iloc(valid_v_idx) if valid_v_idx is not None else None
                else:
                    train_data = self.data.iloc(train_v_idx).select_columns(v)
                    if valid_v_idx is not None:
                        train_v_data = self.data.iloc(valid_v_idx).select_columns(v)
                    else:
                        train_v_data = None

                yield (train_data, train_v_data), outer_valid_data

        return ret_func()

class Experimenter():
    def __init__(self, data, data_names = None, sp = ShuffleSplit(n_splits = 1, random_state=1), sp_v = None, **args):
        self.train_idx_list = list()
        self.valid_idx_list = list()
        data_native = data
        data = wrap(data)
        self.root = data
        split_params = {}

        if data_names is None:
            data_names = data.get_columns()
        for k, v in args.items():
            split_params[k] = unwrap(data.select_columns(v))

        for train_idx, valid_idx in sp.split(data_native, **split_params):
            if sp_v is not None:
                train_data = data.iloc(train_idx)
                train_data_native = unwrap(train_data)

                
                split_params = {'X': train_data_native}
                for k, v in args.items():
                    split_params[k] = unwrap(train_data.select_columns(v))

                self.train_idx_list.append([
                    (train_idx[train_v_idx], train_idx[valid_v_idx])
                    for train_v_idx, valid_v_idx in sp_v.split(**split_params)
                ])
            else:
                self.train_idx_list.append([
                    (train_idx, None)
                ])
            self.valid_idx_list.append(valid_idx)
        self.nodes = {None: RootNode(self, data)}
        self.grps = {}

    def _find_descendants(self, node_name):
        """특정 노드에 의존하는 모든 하위 노드들을 찾음 (BFS)"""
        descendants = set()
        queue = [node_name]

        while queue:
            current = queue.pop(0)

            # 현재 노드에 의존하는 노드들 찾기
            for name, node in self.nodes.items():
                if name is None or name in descendants:
                    continue

                # 이 노드의 edges를 확인
                if hasattr(node, 'edges'):
                    for edge_name, _ in node.edges:
                        if edge_name == current:
                            descendants.add(name)
                            queue.append(name)
                            break

        return descendants

    def _check_cycle(self, node_name, new_edges):
        """특정 노드에 새로운 edges를 추가했을 때 사이클이 생기는지 체크

        Args:
            node_name: 체크할 노드 이름
            new_edges: 추가할 edges 리스트 [(edge_name, var), ...]

        Returns:
            tuple: (has_cycle: bool, cycle_edges: list)
                - has_cycle: 사이클이 있으면 True, 없으면 False
                - cycle_edges: 사이클을 만드는 edge 이름들 리스트
        """
        # node_name의 descendants를 먼저 구함
        descendants = self._find_descendants(node_name)

        cycle_edges = []
        for edge_name, _ in new_edges:
            # Root(None)로의 edge는 사이클을 만들지 않음
            if edge_name is None:
                continue

            # edge_name이 실제 노드인지 확인
            if edge_name not in self.nodes:
                continue

            # edge_name이 node_name의 descendants에 있으면 사이클
            # node_name -> ... -> edge_name (이미 존재)
            # node_name -> edge_name (새로 추가)
            # 이면 node_name -> edge_name -> ... -> node_name 사이클이 생김
            if edge_name in descendants:
                cycle_edges.append(edge_name)

        if cycle_edges:
            return True, cycle_edges
        return False, []

    def _rebuild_node_and_descendants(self, node_name):
        """노드와 그 하위 노드들을 모두 재빌드"""
        # 재빌드할 노드들 찾기
        nodes_to_rebuild = [node_name] + sorted(self._find_descendants(node_name))

        print(f"🔄 Rebuilding nodes: {nodes_to_rebuild}")

        # 토폴로지컬 순서로 재빌드 (의존하는 순서대로)
        for name in nodes_to_rebuild:
            if name in self.nodes and hasattr(self.nodes[name], 'build'):
                print(f"  ├─ Rebuilding '{name}'...")
                self.nodes[name].build()

        print("✅ Rebuild complete!")
    
    def add_grp(self, name, processor = None, edges = list(), X = None, y = None, method = None, parent_grp = None, adapter = 'default', params = None):
        # parent_grp가 문자열이면 grps에서 찾기
        if isinstance(parent_grp, str):
            if parent_grp not in self.grps:
                raise ValueError(f"Parent group '{parent_grp}' not found")
            parent_grp = self.grps.get(parent_grp)

        # NodeGroup 생성
        grp = NodeGroup(self, name, processor=processor, edges=edges, X=X, y=y, method=method, parent_grp=parent_grp, adapter=adapter, params=params)

        # parent의 child_grps에 추가
        if parent_grp is not None:
            parent_grp.child_grps.append(grp)

        # grps 딕셔너리에 등록
        self.grps[name] = grp

        return grp

    def set_grp(self, name, processor = None, edges = None, X = None, y = None, method = None, parent_grp = None, adapter = 'default', params = None):
        if name not in self.grps:
            print(f"⚠️  Group '{name}' not found")
            return

        grp = self.grps[name]

        # parent_grp 변경 처리
        if parent_grp is not None:
            # parent_grp가 문자열이면 grps에서 찾기
            if isinstance(parent_grp, str):
                new_parent = self.grps.get(parent_grp, None)
                if new_parent is None:
                    print(f"⚠️  Parent group '{parent_grp}' not found")
                    return
            else:
                new_parent = parent_grp

            # 이전 parent_grp와 다른 경우
            if grp.parent_grp != new_parent:
                # 이전 parent의 child_grps에서 제거
                if grp.parent_grp is not None:
                    grp.parent_grp.child_grps.remove(grp)

                # 새로운 parent의 child_grps에 추가
                grp.parent_grp = new_parent
                if new_parent is not None:
                    new_parent.child_grps.append(grp)

        # 그룹 속성 업데이트
        if processor is not None:
            grp.processor = processor
        if edges is not None:
            grp.edges = edges if isinstance(edges, list) else [edges]
        if X is not None:
            grp.X = X
        if y is not None:
            grp.y = y
        if method is not None:
            grp.method = method
        if adapter is not None:
            grp.adapter = adapter
        if params is not None:
            grp.params.update(params)

        # 그룹에 속한 노드들 찾기
        if len(grp.nodes) == 0:
            print(f"✅ Group '{name}' updated (no nodes to rebuild)")
            return

        # edges가 업데이트된 경우, 그룹에 속한 노드들의 사이클 체크
        if edges is not None:
            for node_name in grp.nodes:
                if node_name in self.nodes:
                    node = self.nodes[node_name]
                    # 노드의 최종 edges 계산 (그룹 edges + 노드 자체 edges)
                    node_edges = list(node.org_attr['edges']) if node.org_attr else []
                    final_edges = grp.edges + node_edges

                    # 사이클 체크
                    has_cycle, cycle_edges = self._check_cycle(node_name, final_edges)
                    if has_cycle:
                        cycle_info = ", ".join([f"'{e}'" for e in cycle_edges])
                        raise ValueError(f"Cannot update group '{name}': node '{node_name}' would create cycle through edge(s) {cycle_info}")

            print(f"✅ Cycle check passed for all nodes in group '{name}'")

        # 우선순위 알고리즘: BFS로 노드들의 빌드 우선순위 결정
        priorities = {}
        queue = []

        # 변경된 그룹의 노드들을 Root로 우선순위 1 할당
        for node_name in grp.nodes:
            priorities[node_name] = 1
            queue.append((node_name, 1))

        # BFS로 하위 노드들 탐색
        while queue:
            current_node, current_priority = queue.pop(0)

            # 현재 노드에 의존하는 하위 노드들 찾기
            descendants = self._find_descendants(current_node)

            for desc_node in descendants:
                new_priority = current_priority + 1
                # 가장 마지막에 배정된 우선순위가 최종 우선순위
                if desc_node not in priorities or priorities[desc_node] < new_priority:
                    priorities[desc_node] = new_priority
                    queue.append((desc_node, new_priority))

        # 우선순위 순으로 정렬 (낮은 숫자가 먼저)
        sorted_nodes = sorted(priorities.items(), key=lambda x: x[1])

        print(f"🔄 Rebuilding {len(sorted_nodes)} node(s) affected by group '{name}' update")

        # 순서대로 rebuild
        for node_name, priority in sorted_nodes:
            if node_name in self.nodes:
                node = self.nodes[node_name]
                if node.org_attr is not None:
                    print(f"  ├─ Rebuilding '{node_name}' (priority: {priority})...")
                    # org_attr을 사용하여 set_node 재호출
                    org = node.org_attr
                    self.set_node(
                        node_name,
                        grp=node.grp_name,
                        processor=org['processor'],
                        edges=org['edges'],
                        X=org['X'],
                        y=org['y'],
                        method=org['method'],
                        adapter=org['adapter'],
                        rebuild_descendants=False,  # 이미 순서대로 rebuild 중
                        params=org['params']
                    )

        print("✅ Rebuild complete!")

    def remove_grp(self, name):
        if name not in self.grps:
            raise ValueError(f"Group '{name}' not found")

        grp = self.grps[name]

        # child group이 있으면 제거 불가
        if len(grp.child_grps) > 0:
            raise ValueError(f"Cannot remove group '{name}': has {len(grp.child_grps)} child group(s)")

        # 소속 Node가 있으면 제거 불가
        if len(grp.nodes) > 0:
            raise ValueError(f"Cannot remove group '{name}': has {len(grp.nodes)} node(s)")

        # parent의 child_grps에서 제거
        if grp.parent_grp is not None:
            grp.parent_grp.child_grps.remove(grp)

        # grps 딕셔너리에서 제거
        del self.grps[name]

        print(f"✅ Group '{name}' removed")


    def set_node(
        self, name, grp = None, processor = None, edges = list(), X = None, y = None, 
        method = None, rebuild_descendants = True, adapter = 'default', params = None
    ):
        # 기존 노드가 있는지 확인
        is_update = name in self.nodes

        if is_update:
            print(f"⚠️  Updating existing node '{name}'")

        # params 기본값 처리
        if params is None:
            params = {}

        # org_attr 생성 (원본 파라미터 저장)
        org_attr = {
            'processor': processor,
            'edges': edges,
            'X': X,
            'y': y,
            'method': method,
            'adapter': adapter,
            'params': params
        }

        # grp 이름 저장
        grp_name = None
        grp_obj = None

        # grp 처리
        if grp is not None:
            # grp가 문자열이면 grps에서 찾기
            if isinstance(grp, str):
                grp_name = grp
                grp_obj = self.grps.get(grp, None)
                if grp_obj is None:
                    raise ValueError(f"Group '{grp}' not found")
            else:
                grp_name = grp.name
                grp_obj = grp

            # grp의 attrs를 가져와서 기본값으로 사용
            grp_attrs = grp_obj.get_attrs()

            # 파라미터로 넘어온 값이 None이 아니면 override
            if processor is None:
                processor = grp_attrs.get('processor', None)
            if len(grp_attrs['edges']) > 0:
                edges = edges + grp_attrs['edges']
            if X is None:
                X = grp_attrs['X']
            if y is None:
                y = grp_attrs['y']
            if method is None:
                method = grp_attrs.get('method', None)
            if adapter is None:
                adapter = grp_attrs.get('adapter', None)

            # params는 grp의 params를 가져와서 현재 params로 override
            merged_params = {**grp_attrs['params'], **params}
        else:
            merged_params = params

        # processor 체크
        if processor is None:
            raise ValueError(f"Cannot create node '{name}': processor is required")

        # method가 None이면 기본값 설정
        if method is None:
            raise ValueError(f"Cannot create node '{name}': method is required")

        # edges를 리스트로 정규화
        if not isinstance(edges, list):
            edges = [edges]

        # 사이클 체크
        has_cycle, cycle_edges = self._check_cycle(name, edges)
        if has_cycle:
            cycle_info = ", ".join([f"'{e}'" for e in cycle_edges])
            raise ValueError(f"Cannot add node '{name}': would create cycle through edge(s) {cycle_info}")

        node = Node(self, name, processor, edges, X = X, y = y, method = method, grp_name = grp_name, adapter = adapter, org_attr = org_attr, params = merged_params)
        # grp에 노드 추가
        if grp_obj is not None:
            if name not in grp_obj.nodes:
                grp_obj.nodes.append(name)

        # 기존 노드를 업데이트한 경우, 하위 노드들도 재빌드
        if is_update and rebuild_descendants:
            descendants = self._find_descendants(name)
            if descendants:
                print(f"  └─ Found {len(descendants)} dependent node(s): {sorted(descendants)}")
                self._rebuild_node_and_descendants(name)

        # 그룹이 변경된 경우 이전 그룹에서 노드 제거
        if is_update and self.nodes[name].grp_name != grp_name:
            old_grp_name = self.nodes[name].grp_name
            if old_grp_name is not None and old_grp_name in self.grps:
                old_grp = self.grps[old_grp_name]
                if name in old_grp.nodes:
                    old_grp.nodes.remove(name)
                    print(f"  ├─ Removed '{name}' from group '{old_grp_name}'")
            if grp_name is not None:
                print(f"  └─ Moved '{name}' to group '{grp_name}'")

        self.nodes[name] = node
        return node

    def rebuild_all(self):
        """모든 노드를 재빌드 (Root 제외)"""
        print("🔄 Rebuilding all nodes...")

        for name, node in self.nodes.items():
            if name is not None and hasattr(node, 'build'):
                print(f"  ├─ Rebuilding '{name}'...")
                node.build()

        print("✅ All nodes rebuilt!")
    
    def get_data(self, idx, edges):
        def ret_data_func(data_list):
            for z in zip(*data_list):
                train_sub, valid_sub, outer_valid_sub = list(), list(), list()
                for (train_data, train_v_data), outer_valid_data in z:
                    train_sub.append(train_data)
                    if train_v_data is not None:
                        valid_sub.append(train_v_data)
                    outer_valid_sub.append(outer_valid_data)

                # DataWrapper의 concat 사용
                if len(valid_sub) > 0:
                    train_concat = type(train_sub[0]).concat(train_sub, axis=1)
                    valid_concat = type(valid_sub[0]).concat(valid_sub, axis=1)
                    outer_concat = type(outer_valid_sub[0]).concat(outer_valid_sub, axis=1)
                    yield (train_concat, valid_concat), outer_concat
                else:
                    train_concat = type(train_sub[0]).concat(train_sub, axis=1)
                    outer_concat = type(outer_valid_sub[0]).concat(outer_valid_sub, axis=1)
                    yield (train_concat, None), outer_concat

        data_list = list()
        for node_name, var in edges:
            data_list.append(self.nodes[node_name].get_data(idx, var))
        return ret_data_func(data_list)
    
    def split(self, edges):
        for idx in range(len(self.train_idx_list)):
            yield self.get_data(idx, edges)

    def get_node_info(self):
        """노드들의 정보를 출력"""
        print("📊 Experiment Pipeline Summary")
        print("=" * 50)

        for name, node in self.nodes.items():
            if name is None:
                print(f"Root Node: {type(self.root).__name__}")
            else:
                processor_name = node.processor.__name__
                edges_info = ", ".join([
                    f"{n or 'Root'}{f'[{v}]' if v else ''}"
                    for n, v in node.edges
                ])
                print(f"\nNode: '{name}'")
                print(f"  ├─ Processor: {processor_name}")
                print(f"  ├─ Method: {node.method}")
                print(f"  ├─ Edges: {edges_info}")

                descendants = self._find_descendants(name)
                if descendants:
                    print(f"  └─ Descendants: {sorted(descendants)}")

    def to_mermaid(self, max_depth=None, direction='TD'):
        """실험 구조를 Mermaid markdown으로 반환

        Args:
            max_depth: 최대 표시 깊이 (None이면 무제한)
            direction: 그래프 방향 ('TD': Top-Down, 'LR': Left-Right)
        """
        # 노드 개수 계산 함수
        def count_nodes_in_group(grp):
            count = len(grp.nodes)
            for child_grp in grp.child_grps:
                count += count_nodes_in_group(child_grp)
            return count

        # 1. Node 단위 우선순위 생성 (BFS)
        node_priorities = {}
        queue = [('Root', 1)]

        while queue:
            current_node, priority = queue.pop(0)

            # 이미 더 낮은 우선순위(더 상위)가 할당되었으면 스킵
            if current_node in node_priorities:
                continue

            node_priorities[current_node] = priority

            # current_node를 edge로 가지는 child 노드들 찾기
            for name, node in self.nodes.items():
                if name is not None:
                    for edge_name, _ in node.edges:
                        if (current_node == 'Root' and edge_name is None) or (edge_name == current_node):
                            # child 노드 발견
                            if name not in node_priorities:
                                queue.append((name, priority + 1))

        # 2. Group 단위 우선순위 생성 (포함된 노드 중 가장 낮은 우선순위 = 가장 상위)
        grp_priorities = {}
        for grp_name, grp in self.grps.items():
            if len(grp.nodes) > 0:
                grp_priorities[grp_name] = min(node_priorities.get(node_name, float('inf')) for node_name in grp.nodes)
            else:
                grp_priorities[grp_name] = float('inf')

        # 3. 소속 그룹이 없는 노드와 최상위 그룹 수집
        grouped_nodes = set()
        for grp in self.grps.values():
            grouped_nodes.update(grp.nodes)

        ungrouped_nodes = [name for name in self.nodes.keys() if name is not None and name not in grouped_nodes]
        top_level_items = []

        # 최상위 그룹 (parent_grp가 None인 그룹)
        for grp_name, grp in self.grps.items():
            if grp.parent_grp is None:
                top_level_items.append(('group', grp))

        # 소속 그룹이 없는 노드들
        for node_name in ungrouped_nodes:
            if node_name in self.nodes:
                top_level_items.append(('node', self.nodes[node_name]))

        # 우선순위로 정렬
        def get_priority(item):
            item_type, obj = item
            if item_type == 'group':
                return grp_priorities.get(obj.name, float('inf'))
            else:
                return node_priorities.get(obj.name, float('inf'))

        top_level_items.sort(key=get_priority)

        # 4. Mermaid 생성
        lines = []
        lines.append("```mermaid")
        lines.append(f"graph {direction}")
        lines.append("")

        # Root 노드
        lines.append("    Root([Root])")
        lines.append("    style Root fill:#fff9c4,stroke:#f57c00,stroke-width:3px")
        lines.append("")

        # Recursive 함수로 그룹과 노드 생성
        def render_group(grp, indent=4, current_depth=1):
            indent_str = " " * indent
            result = []
            result.append(f"{indent_str}subgraph grp_{grp.name}[\"{grp.name}\"]")

            # max_depth에 도달했으면 노드 개수만 표시
            if max_depth is not None and current_depth >= max_depth:
                node_count = count_nodes_in_group(grp)
                result.append(f"{indent_str}    grp_{grp.name}_count[\"{node_count} node(s)\"]")
                result.append(f"{indent_str}    style grp_{grp.name}_count fill:#f5f5f5,stroke:#9e9e9e,stroke-dasharray: 5 5")
            else:
                # 그룹 내부의 child_grps와 nodes 수집
                items = []
                for child_grp in grp.child_grps:
                    items.append(('group', child_grp))
                for node_name in grp.nodes:
                    if node_name in self.nodes:
                        items.append(('node', self.nodes[node_name]))

                # 우선순위로 정렬
                items.sort(key=get_priority)

                # 렌더링
                for item_type, obj in items:
                    if item_type == 'group':
                        result.extend(render_group(obj, indent + 4, current_depth + 1))
                    else:
                        node_name = obj.name
                        result.append(f"{indent_str}    node_{node_name}[\"{node_name}\"]")
                        result.append(f"{indent_str}    style node_{node_name} fill:#c8e6c9,stroke:#388e3c,stroke-width:2px")

            result.append(f"{indent_str}end")
            result.append(f"{indent_str}style grp_{grp.name} fill:#e3f2fd,stroke:#1976d2,stroke-width:2px")
            return result

        # Top-level items 렌더링
        for item_type, obj in top_level_items:
            if item_type == 'group':
                # top-level 그룹은 depth 1부터 시작
                if max_depth is None or max_depth >= 1:
                    lines.extend(render_group(obj, indent=4, current_depth=1))
                    lines.append("")
            else:
                # top-level 노드도 depth 1
                if max_depth is None or max_depth >= 1:
                    node_name = obj.name
                    lines.append(f"    node_{node_name}[\"{node_name}\"]")
                    lines.append(f"    style node_{node_name} fill:#c8e6c9,stroke:#388e3c,stroke-width:2px")
                    lines.append("")

        # Edge 연결 알고리즘
        # 1. 각 노드가 속한 최상위 노드를 담는 딕셔너리
        node_to_top = {}
        for item_type, obj in top_level_items:
            if item_type == 'group':
                # 그룹에 속한 모든 노드 수집 (recursive)
                def collect_nodes_in_group(grp):
                    nodes = []
                    for node_name in grp.nodes:
                        nodes.append(node_name)
                    for child_grp in grp.child_grps:
                        nodes.extend(collect_nodes_in_group(child_grp))
                    return nodes

                nodes_in_grp = collect_nodes_in_group(obj)
                for node_name in nodes_in_grp:
                    node_to_top[node_name] = ('group', obj.name)
            else:
                node_to_top[obj.name] = ('node', obj.name)

        # 2. 상위 노드에서 하위 노드를 DFS 탐색하며 incoming 연결 수집
        top_node_incoming = {}  # top_node -> set of incoming top_nodes

        def dfs_collect_incoming(top_item_type, top_item_name):
            incoming = set()

            # top_item에 속한 모든 노드 찾기
            if top_item_type == 'group':
                grp = self.grps[top_item_name]
                def collect_nodes_in_group(grp):
                    nodes = []
                    for node_name in grp.nodes:
                        nodes.append(node_name)
                    for child_grp in grp.child_grps:
                        nodes.extend(collect_nodes_in_group(child_grp))
                    return nodes
                nodes = collect_nodes_in_group(grp)
            else:
                nodes = [top_item_name]

            # 각 노드의 edges 확인
            for node_name in nodes:
                if node_name in self.nodes:
                    node = self.nodes[node_name]
                    for edge_name, edge_var in node.edges:
                        if edge_name is None:
                            # Root 연결
                            incoming.add(('root', 'Root'))
                        elif edge_name in node_to_top:
                            # edge 노드의 최상위 노드 찾기
                            edge_top = node_to_top[edge_name]
                            # 같은 top node 내부 연결은 제외
                            if not (top_item_type == edge_top[0] and top_item_name == edge_top[1]):
                                incoming.add(edge_top)

            return incoming

        # 각 top-level item의 incoming 수집
        for item_type, obj in top_level_items:
            if item_type == 'group':
                key = ('group', obj.name)
            else:
                key = ('node', obj.name)
            top_node_incoming[key] = dfs_collect_incoming(item_type, obj.name if item_type == 'group' else obj.name)

        # 3. Edge 출력
        edges_set = set()
        for target, incoming_set in top_node_incoming.items():
            target_type, target_name = target
            target_id = f"grp_{target_name}" if target_type == 'group' else f"node_{target_name}"

            for source_type, source_name in incoming_set:
                if source_type == 'root':
                    source_id = "Root"
                elif source_type == 'group':
                    source_id = f"grp_{source_name}"
                else:
                    source_id = f"node_{source_name}"

                edges_set.add((source_id, target_id))

        for source, target in sorted(edges_set):
            lines.append(f"    {source} --> {target}")

        lines.append("```")
        lines.append("")

        # Splitter 정보
        splitter_info = f"Experimenter (n_splits={len(self.train_idx_list)})"
        if max_depth is not None:
            splitter_info += f", max_depth={max_depth}"
        lines.append(f"**{splitter_info}**")

        return "\n".join(lines)

    def node_to_mermaid(self, node_name, direction='TD', show_params=False):
        """특정 노드까지의 연결 구조를 Mermaid markdown으로 반환

        Args:
            node_name: 대상 노드 이름
            direction: 그래프 방향 ('TD': Top-Down, 'LR': Left-Right)
            show_params: True이면 노드의 파라미터 정보를 표시 (default: False)
        """
        if node_name not in self.nodes or node_name is None:
            return f"Node '{node_name}' not found"

        # Root에서 node_name까지의 경로 찾기 (BFS)
        def find_paths_to_node(target):
            paths = []
            queue = [(['Root'], set(['Root']))]

            while queue:
                path, visited = queue.pop(0)
                current = path[-1]

                # target에 도달했으면 경로 저장
                if current == target:
                    paths.append(path[:])
                    continue

                # current를 edge로 가지는 노드들 찾기
                for name, node in self.nodes.items():
                    if name is not None and name not in visited:
                        for edge_name, _ in node.edges:
                            if (current == 'Root' and edge_name is None) or (edge_name == current):
                                new_path = path + [name]
                                new_visited = visited | {name}
                                queue.append((new_path, new_visited))
                                break

            return paths

        paths = find_paths_to_node(node_name)

        if not paths:
            return f"No path from Root to '{node_name}'"

        # Mermaid 생성
        lines = []
        lines.append("```mermaid")
        lines.append(f"graph {direction}")
        lines.append("")

        # Root 노드
        lines.append("    Root([Root])")
        lines.append("    style Root fill:#fff9c4,stroke:#f57c00,stroke-width:3px")
        lines.append("")

        # 경로에 포함된 모든 노드 수집
        all_nodes = set()
        for path in paths:
            all_nodes.update(path)
        all_nodes.discard('Root')

        # 각 노드를 subgraph로 생성
        for name in sorted(all_nodes):
            if name in self.nodes:
                node = self.nodes[name]

                # subgraph의 title은 항상 노드 이름만
                lines.append(f"    subgraph node_{name}[\"{name}\"]")

                if show_params:
                    # 파라미터 정보 포맷팅
                    processor_name = node.processor.__name__ if node.processor else 'None'

                    info_parts = ["<table>"]
                    info_parts.append(f"<tr><td align='left'><b>processor</b></td><td align='left'>{processor_name}</td></tr>")
                    info_parts.append(f"<tr><td align='left'><b>method</b></td><td align='left'>{node.method}</td></tr>")

                    # params 정보
                    if node.params:
                        for key, value in node.params.items():
                            value_str = str(value)
                            if len(value_str) > 40:
                                value_str = value_str[:37] + '...'
                            info_parts.append(f"<tr><td align='left'><b>{key}</b></td><td align='left'>{value_str}</td></tr>")
                        info_parts.append("</table>")
                    params_content = "".join(info_parts)
                    lines.append(f"        {name}_info[\"{params_content}\"]")
                else:
                    # show_params가 False면 빈 더미 노드
                    lines.append(f"        {name}_dummy[ ]")
                    lines.append(f"        style {name}_dummy fill:none,stroke:none")

                lines.append(f"    end")

                # target 노드는 다른 색으로 표시
                if name == node_name:
                    lines.append(f"    style node_{name} fill:#ffcdd2,stroke:#c62828,stroke-width:3px")
                else:
                    lines.append(f"    style node_{name} fill:#c8e6c9,stroke:#388e3c,stroke-width:2px")
                lines.append("")

        # 경로상의 엣지만 표시
        edges_set = set()
        for path in paths:
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                if source == 'Root':
                    edges_set.add(("Root", f"node_{target}"))
                else:
                    edges_set.add((f"node_{source}", f"node_{target}"))

        for source, target in sorted(edges_set):
            lines.append(f"    {source} --> {target}")

        lines.append("```")
        lines.append("")
        lines.append(f"**Path from Root to '{node_name}' ({len(paths)} path(s) found)**")

        return "\n".join(lines)