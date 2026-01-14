import uuid
import adapter
from processor import TransformProcessor, PredictProcessor, resolve_columns

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

    def _unload_cache(self):
        self.cache_idx = -1
        self.cache = None


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
