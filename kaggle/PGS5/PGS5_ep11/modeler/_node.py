import uuid
from ._adapter import get_adapter
from ._node_processor import TransformProcessor, PredictProcessor, resolve_columns
import numpy as np

class NodeGroup():
    def __init__(
        self, experimenter, name, processor = None, edges = list(), X = None, y = None, 
        method = 'transform', parent_grp = None, adapter = 'default', params = None
    ):
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

    def _unload_cache(self):
        self.cache_idx = -1
        self.cache = None
        self.cache_v_param = None
        self.cache_idx_v = -1
        self.cache_v = None
        self.cache_v_param_v = None
        self.cache_idx_t = -1
        self.cache_t = None
        self.cache_t_param_v = None


    def build(self):
        if self.method in ['transform', 'predict', 'predict_proba']:
            self._fit()
        elif self.method in ['fit_transform', 'fit_predict']:
            self._fit_process()
        else:
            raise ValueError(f"Unknown processor_type: {self.method}")

    def _fit(self):
        self._unload_cache()
        self.objs_ = list()

        # adapter 인스턴스 가져오기
        if self.adapter == 'default':
            adapter_ = get_adapter(self.processor)
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
                    'build_id': str(uuid.uuid4()),
                    'fit_time': elapsed_time,
                    'train_shape': train_t.get_shape() if train_t is not None else None,
                    'train_v_shape': train_v.get_shape() if train_v is not None else None
                }
                sub.append((obj, None, info))
            self.objs_.append(sub)

        # 완료 메시지 출력
        print(f"\r[{self.name}] Building: {total_folds}/{total_folds} (100%) ✓ Complete")

    def _fit_process(self):
        self._unload_cache()
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
                    'build_id': str(uuid.uuid4()),
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
        if self.cache_idx == idx and self.cache_v_param == v and self.cache is not None:
            def ret_func():
                for i in self.cache:
                    yield i
            return ret_func()
        it = self.experimenter.get_data(idx, self.edges)
        sub = self.objs_[idx]
        if self.use_cache:
            self._unload_cache()
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
            self.cache_v_param = v
        return ret_func()

    def get_data_train(self, idx, v=None):
        if self.cache_idx_t == idx and self.cache_t_param_v == v and self.cache_t is not None:
            def ret_func():
                for i in self.cache_t:
                    yield i
            return ret_func()
        it = self.experimenter.get_data_train(idx, self.edges)
        sub = self.objs_[idx]
        if self.use_cache:
            self._unload_cache()
        def ret_func():
            if self.use_cache:
                self.cache_t = list()
            else:
                self.cache_t = None
            for (train, train_v), (obj, train_, info) in zip(it, sub):
                train_result = obj.process(train) if train_ is None else train_
                if train_v is not None:
                    train_v_result = obj.process(train_v)
                else:
                    train_v_result = None
                # 필요하면 컬럼 필터링
                if v is not None:
                    X = resolve_columns(train_result, v, org_X=obj.X_)
                    train_result = train_result.select_columns(X)
                    if train_v_result is not None:
                        train_v_result = train_v_result.select_columns(X)
                if self.cache_t is not None:
                    self.cache_t.append((train_result, train_v_result))
                yield train_result, train_v_result
        if self.use_cache:
            self.cache_idx_t = idx
            self.cache_t_param_v = v
        return ret_func()
    
    def get_data_valid(self, idx, v=None):
        """외부 검증 데이터에 대한 처리 결과 iterator

        Args:
            idx: outer fold 인덱스
            v: 선택할 컬럼 (None이면 전체)

        Yields:
            valid_result: 각 inner fold 모델로 처리된 외부 검증 데이터 결과
        """
        if self.cache_idx_v == idx and self.cache_v_param_v == v and self.cache_v is not None:
            def ret_func():
                for i in self.cache_v:
                    yield i
            return ret_func()

        it = self.experimenter.get_data_valid(idx, self.edges)
        sub = self.objs_[idx]
        if self.use_cache:
            self._unload_cache()
        def ret_func():
            if self.use_cache:
                self.cache_v = list()
            else:
                self.cache_v = None

            for valid, (obj, train_, info) in zip(it, sub):
                # valid data 처리 (외부 fold의 valid)
                valid_result = obj.process(valid)

                # 필요하면 컬럼 필터링
                if v is not None:
                    X = resolve_columns(valid_result, v, org_X=obj.X_)
                    valid_result = valid_result.select_columns(X)

                if self.cache_v is not None:
                    self.cache_v.append(valid_result)
                yield valid_result

        if self.use_cache:
            self.cache_idx_v = idx
            self.cache_v_param_v = v
        return ret_func()
    
    
class RootNode():
    def __init__(self, experimenter, data):
        self.experimenter = experimenter
        self.data = data

    def get_data(self, idx, v=None):
        outer_valid_data = self.data.iloc(self.experimenter.valid_idx_list[idx])
        if v is not None:
            outer_valid_data = outer_valid_data.select_columns(v)
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

    def get_data_train(self, idx, v=None):
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

                yield train_data, train_v_data

        return ret_func()
    
    def get_data_valid(self, idx, v=None):
        """외부 검증 데이터만 반환하는 iterator

        Args:
            idx: outer fold 인덱스
            v: 선택할 컬럼 (None이면 전체)

        Yields:
            outer_valid_data: 외부 검증 데이터 (각 inner fold마다 동일한 데이터)
        """
        outer_valid_data = self.data.iloc(self.experimenter.valid_idx_list[idx])
        if v is not None:
            outer_valid_data = outer_valid_data.select_columns(v)

        def ret_func():
            for _ in self.experimenter.train_idx_list[idx]:
                yield outer_valid_data

        return ret_func()
