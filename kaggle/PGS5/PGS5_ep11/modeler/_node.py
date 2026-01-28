import uuid
import pickle as pkl
from .adapter import get_adapter
from ._node_processor import TransformProcessor, PredictProcessor, resolve_columns
import numpy as np
import pandas as pd
import os
import shutil
import time

class NodeGroup():
    def __init__(
        self, experimenter, name, role, processor = None, edges = list(), X = None, y = None,
        method = 'transform', parent_grp = None, adapter = 'default', params = None
    ):
        self.experimenter = experimenter
        self.name = name
        self.role = role
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

    @property
    def path(self):
        """그룹의 디렉터리 경로 (Experimenter.path 기준)"""
        if self.experimenter.path is None:
            return None

        # 부모 그룹 경로 수집
        path_parts = [self.name]
        current = self.parent_grp
        while current is not None:
            path_parts.insert(0, current.name)
            current = current.parent_grp

        return self.experimenter.path / '/'.join(path_parts)

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

    def save_info(self):
        info = {
            'name': self.name,
            'role': self.role,
            'processor': self.processor,
            'edges': self.edges,
            'X': self.X,
            'y': self.y,
            'method': self.method,
            'params': self.params,
            'parent_grp': self.parent_grp.name if self.parent_grp is not None else None,
            'child_grps': [grp.name for grp in self.child_grps],
            'adapter': self.adapter
        }
        filepath = self.path / f'__grp.pkl'
        with open(filepath, 'wb') as f:
            pkl.dump(info, f)
        return filepath

    def load_info(self):
        """저장된 그룹 정보를 불러와서 어트리뷰트에 설정 (파일명: __<name>.pkl)

        Returns:
            dict: 그룹 정보 딕셔너리
        """
        filepath = self.path / f'__grp.pkl'
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Group info file not found: {filepath}")
        with open(filepath, 'rb') as f:
            info = pkl.load(f)

        # 어트리뷰트에 설정
        self.role = info['role']
        self.processor = info['processor']
        self.edges = info['edges']
        self.X = info['X']
        self.y = info['y']
        self.method = info['method']
        self.params = info['params']
        self.adapter = info['adapter']

        return info

    @classmethod
    def load(cls, experimenter, name, parent_grp=None):
        """저장된 그룹 정보를 불러와서 NodeGroup 인스턴스 생성

        Args:
            experimenter: Experimenter 인스턴스
            name: 그룹 이름
            parent_grp: 부모 그룹 객체 (None이면 최상위 그룹)

        Returns:
            NodeGroup: 복원된 NodeGroup 인스턴스
        """
        # 경로 계산 (parent_grp가 있으면 그 아래, 없으면 experimenter.path 바로 아래)
        if parent_grp is not None:
            filepath = parent_grp.path / name / '__grp.pkl'
        else:
            filepath = experimenter.path / name / '__grp.pkl'

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Group info file not found: {filepath}")

        with open(filepath, 'rb') as f:
            info = pkl.load(f)

        # NodeGroup 인스턴스 생성 (기본값으로 생성 후 속성 설정)
        grp = cls(
            experimenter=experimenter,
            name=name,
            role=info['role'],
            processor=info['processor'],
            edges=info['edges'],
            X=info['X'],
            y=info['y'],
            method=info['method'],
            parent_grp=parent_grp,
            adapter=info['adapter'],
            params=info['params']
        )

        # parent의 child_grps에 추가
        if parent_grp is not None:
            parent_grp.child_grps.append(grp)

        return grp

class Node():
    def __init__(
        self, experimenter, name, processor, edges, grp, X = None, y = None, method = 'transform',
        use_cache = True, org_attr = None, adapter = 'default', params = None
    ):
        self.experimenter = experimenter
        self.name = name
        self.grp = grp  # 속한 그룹
        self.org_attr = org_attr  # 원본 속성 (processor, edges, X, y, method, params)
        self.processor = processor
        self.method = method
        self.edges = edges
        self.params = params if params is not None else {}
        self.X = X
        self.y = y
        self.use_cache = use_cache
        self.output_edges = []  # 이 노드를 입력으로 사용하는 노드들의 이름
        # adapter 인스턴스 가져오기
        if adapter == 'default':
            self.adapter_ = get_adapter(self.processor)
        else:
            self.adapter_ = self.adapter
        self.status = None
        self._unload_cache()
        self.save_info()

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
        self.status = "built"

    def start_build(self):
        self.initialize()
        self.objs_ = list()
        if not os.path.isdir(self.path):
                os.makedirs(self.path, exist_ok = True)
    
    def _build_idx(self, idx):
        if self.method in ['transform', 'predict', 'predict_proba']:
            bobj = self._build_obj(self.experimenter.get_data(idx, self.edges), False)
        elif self.method in ['fit_transform', 'fit_predict']:
            bobj = self._build_obj(self.experimenter.get_data(idx, self.edges), True)
        else:
            raise ValueError(f"Unknown processor_type: {self.method}")
        return bobj
    
    def build_idx(self, idx):
        if idx != len(self.objs_):
            raise RuntimeError(f"{self.name}: Build sequence is not valid")

        filename = self.path / ('obj' + str(idx) + '.pkl')
        if self.status == "built":
            with open(filename, 'rb') as f:
                bobj = pkl.load(f)
        else:
            bobj = self._build_idx(idx)
            with open(filename, 'wb') as f:
                pkl.dump(bobj, f)
        self.objs_.append(bobj)

    def end_build(self):
        self.status = "built"
        self.save_info()

    def _build_sub(self, train_t, train_v, fit_process):
        if self.method in ['transform', 'fit_transform']:
            obj = TransformProcessor(self, self.processor, X = self.X, y = self.y, adapter = self.adapter_, **self.params)
        else:
            obj = PredictProcessor(self, self.processor, X = self.X, y = self.y, method = self.method, adapter = self.adapter_, **self.params)

        start_time = time.time()
        if fit_process:
            result = obj.fit_process(train_t, train_v)
        else:
            result = None
            obj.fit(train_t, train_v)
        elapsed_time = time.time() - start_time

        info = {
            'build_id': str(uuid.uuid4()),
            'fit_time': elapsed_time,
            'train_shape': train_t.get_shape() if train_t is not None else None,
            'train_v_shape': train_v.get_shape() if train_v is not None else None
        }
        return obj, result, info

    def _build_obj(self, train, fit_process):
        sub = list()
        for (train_t, train_v), _ in train:
            sub.append(self._build_sub(train_t, train_v, fit_process))
        return sub

    def start_experiment(self):
        if self.status == "finalized":
            raise RuntimeError("")
        else:
            if not os.path.isdir(self.path):
                os.makedirs(self.path, exist_ok = True)
    
    def experiment(self, idx, include_output = True, finalize = False):
        if self.grp.role != 'exp':
            raise RuntimeError("")
        filename = self.path / ('obj' + str(idx) + '.pkl')
        if self.status == "built":
            with open(filename, 'rb') as f:
                objs = pkl.load(f)
        elif self.status == "finalized":
            raise RuntimeError("")
        
        objs = self._build_idx(idx)
        ret = list()
        it = self.experimenter.get_data(idx, self.edges)
        result_list = list()
        for ((train_t, train_v), valid), (obj, train_, spec) in zip(it, objs):
            sub_result = {'spec': spec, 'object': obj}
            if include_output:
                if train_ is None:
                    train_result = obj.process(train_t)
                else:
                    train_result = train_
                if train_v is not None:
                    train_v_result = obj.process(train_v)
                sub_result['output_train'] = (train_result, train_v_result)
                sub_result['output_valid'] = obj.process(valid)
            yield sub_result
        if self.status is None and (not finalize):
            with open(filename, 'wb') as f:
                pkl.dump(objs, f)
    
    def end_experiment(self, finalize = False):
        if finalize:
            self.status = "finalized"
        else:
            self.status = "built"
        self.save_info()

    def finalize(self):
        if self.status != 'built':
            raise RuntimeError("")

        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
        self.status = "finalized"
        self.save_info()
    
    def get_exp_obj(self, idx):
        filename = self.path / ('obj' + str(idx) + '.pkl')
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                objs = pkl.load(f)
            return objs
        else:
            raise RuntimeError("")

    def adhoc(self, idx, results):
        return list(
            self.experiment(idx, results)
        )

    @property
    def path(self):
        return self.grp.path / self.name

    def save_info(self):
        """노드 정보를 pickle로 저장 (파일명: __<name>.pkl)

        Returns:
            Path: 저장된 파일 경로
        """
        info = {
            'name': self.name,
            'grp': self.grp.name if self.grp is not None else None,
            'org_attr': self.org_attr,
            'processor': self.processor,
            'method': self.method,
            'edges': self.edges,
            'params': self.params,
            'X': self.X,
            'y': self.y,
            'use_cache': self.use_cache,
            'adapter': self.adapter_,
            'status': self.status
        }
        filepath = self.grp.path / f'__{self.name}.pkl'
        with open(filepath, 'wb') as f:
            pkl.dump(info, f)
        return filepath

    def load_info(self):
        """저장된 노드 정보를 불러와서 어트리뷰트에 설정 (파일명: __<name>.pkl)

        Returns:
            dict: 노드 정보 딕셔너리
        """
        filepath = self.grp.path / f'__{self.name}.pkl'
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Node info file not found: {filepath}")
        with open(filepath, 'rb') as f:
            info = pkl.load(f)

        # 어트리뷰트에 설정
        self.org_attr = info['org_attr']
        self.processor = info['processor']
        self.method = info['method']
        self.edges = info['edges']
        self.params = info['params']
        self.X = info['X']
        self.y = info['y']
        self.use_cache = info['use_cache']
        self.adapter_ = info['adapter']
        self.status = info['status']

        return info

    @classmethod
    def load(cls, experimenter, grp, name):
        """저장된 노드 정보를 불러와서 Node 인스턴스 생성

        Args:
            experimenter: Experimenter 인스턴스
            grp: NodeGroup 인스턴스
            name: 노드 이름

        Returns:
            Node: 복원된 Node 인스턴스
        """
        filepath = grp.path / f'__{name}.pkl'
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Node info file not found: {filepath}")

        with open(filepath, 'rb') as f:
            info = pkl.load(f)

        # Node 인스턴스 생성 (save_info 호출을 피하기 위해 빈 객체 생성 후 속성 설정)
        node = object.__new__(cls)
        node.experimenter = experimenter
        node.name = name
        node.grp = grp
        node.org_attr = info['org_attr']
        node.processor = info['processor']
        node.method = info['method']
        node.edges = info['edges']
        node.params = info['params']
        node.X = info['X']
        node.y = info['y']
        node.use_cache = info['use_cache']
        node.adapter_ = info['adapter']
        node.status = info['status']
        if node.status == 'built' and grp.role == 'pipe':
            node.objs_ = list()
            for idx in range(experimenter.get_n_splits()):
                filename = node.path / ('obj' + str(idx) + '.pkl')
                with open(filename, 'rb') as f:
                    bobj = pkl.load(f)
                node.objs_.append(bobj)
        node.output_edges = []  # edges 복원 후 재구성됨
        node._unload_cache()

        # grp.nodes에 추가
        if name not in grp.nodes:
            grp.nodes.append(name)

        return node

    def remove(self):
        if os.path.isdir(path):
            shutil.rmtree(path)
        filepath = self.grp.path / f'__{self.name}.pkl'
        if os.path.isfile(filepath):
            os.remove(filepath)
        
    def initialize(self):
        if self.status is None:
            return
        path = self.path
        if os.path.isdir(path):
            shutil.rmtree(path)
        self._unload_cache()
        self.status = None
        self.objs_ = None
        self.save_info()

    def _fit(self):
        self.objs_ = list()
        # 전체 검증 수 계산
        total_folds = len(self.experimenter.train_idx_list)
        current = 0

        for train in self.experimenter.split(self.edges):
            current += 1
            percentage = int(current * 100 / total_folds)
            print(f"\r[{self.name}] Building: {current}/{total_folds} ({percentage}%)", end='', flush=True)
            self.objs_.append(self._build_obj(train, False))

        # 완료 메시지 출력
        print(f"\r[{self.name}] Built: {total_folds}/{total_folds} (100%) ✓ Complete")

    def _fit_process(self):
        self._unload_cache()
        self.objs_ = list()

        # 전체 검증 수 계산
        total_folds = len(self.experimenter.train_idx_list)
        current = 0

        for train in self.experimenter.split(self.edges):
            current += 1
            percentage = int(current * 100 / total_folds)
            print(f"\r[{self.name}] Building: {current}/{total_folds} ({percentage}%)", end='', flush=True)
            self.objs_.append(self._build_obj(train, True))

        # 완료 메시지 출력
        print(f"\r[{self.name}] Built: {total_folds}/{total_folds} (100%) ✓ Complete")

    def get_data(self, idx, v = None):
        if self.grp.role != 'pipe':
            raise RuntimeError(f"Cannot get_data as {self.name} node is included pipeline group.")
        if self.objs_ is None:
            raise RuntimeError(f"{self.name} is not built")
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
        self.output_edges = []  # 이 노드를 입력으로 사용하는 노드들의 이름

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