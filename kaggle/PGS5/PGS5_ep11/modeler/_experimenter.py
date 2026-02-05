import re
import os
import uuid
import pickle as pkl
import shutil
import traceback
import warnings
from pathlib import Path

import pandas as pd

from sklearn.model_selection import ShuffleSplit

from ._data_wrapper import wrap, unwrap
from ._expobj import HeadObj, StageObj
from ._describer import desc_spec, desc_node_vars
from ._metric import Metric
from ._stacking import Stacking
from ._logger import DefaultLogger

from ._pipeline import Pipeline
from ._node_processor import resolve_columns

class DataCache():
    def __init__(self):
        self.cache_dic = {}

    def get_data(self, node, typ, idx, v):
        key = (node, typ, idx, tuple(v) if type(v) == list else None)
        if key in self.cache_dic:
            def ret_func():
                for i in self.cache_dic[key]:
                    yield i
            return ret_func()
        else:
            None
    def put_data(self, node, typ, idx, v, data):
        key = (node, typ, idx, tuple(v) if type(v) == list else None)
        self.cache_dic[key] = data

class Experimenter():
    def __init__(
            self, data, path, data_names = None, sp = ShuffleSplit(n_splits=1, random_state=1), sp_v=None, 
            splitter_params=None, title=None, data_key=None, logger = DefaultLogger(level=['info', 'progress'])
        ):
        self.logger = logger
        self.path = Path(path)
        if not os.path.exists(path):
            self.path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"📁 Created directory: {self.path}")
        self.train_idx_list = list()
        self.valid_idx_list = list()
        data_native = data
        self.data = wrap(data)
        # 실험 타이틀 저장
        self.title = title

        # data 식별자 (load 시 검증용)
        self.data_key = data_key

        # splitter 설정 저장
        self.sp = sp
        self.sp_v = sp_v
        self.splitter_params = splitter_params if splitter_params is not None else {}
        self.exp_id = str(uuid.uuid4())

        split_params = {}

        if data_names is None:
            data_names = self.data.get_columns()
        for k, v in self.splitter_params.items():
            split_params[k] = unwrap(self.data.select_columns(v))

        for train_idx, valid_idx in sp.split(data_native, **split_params):
            if sp_v is not None:
                train_data = self.data.iloc(train_idx)
                train_data_native = unwrap(train_data)

                inner_split_params = {'X': train_data_native}
                for k, v in self.splitter_params.items():
                    inner_split_params[k] = unwrap(train_data.select_columns(v))

                self.train_idx_list.append([
                    (train_idx[train_v_idx], train_idx[valid_v_idx])
                    for train_v_idx, valid_v_idx in sp_v.split(**inner_split_params)
                ])
            else:
                self.train_idx_list.append([
                    (train_idx, None)
                ])
            self.valid_idx_list.append(valid_idx)

        self.pipeline = Pipeline()
        self.node_objs = {}
        self.cache = DataCache()
        self.grps = {}
        self.metric = {}
        self.stacking = {}
        self.status = "open"

    def _check_open(self):
        """상태가 open인지 확인하고, 아니면 에러 발생"""
        if self.status != "open":
            raise RuntimeError(f"Experimenter is '{self.status}'. Only 'open' status allows modifications.")

    def open(self):
        """Experimenter를 open 상태로 변경"""
        self.status = "open"
        self._save()
        self.logger.info("Experimenter status changed to 'open'")

    def close(self):
        """Experimenter를 close 상태로 변경"""
        self.status = "close"
        self._save()
        self.logger.info("Experimenter status changed to 'close'")

    @staticmethod
    def create(data, path, data_names=None, sp=ShuffleSplit(n_splits=1, random_state=1), sp_v=None, splitter_params=None, title=None, data_key=None,
            logger = DefaultLogger(level=['info', 'progress'])):
        
        if os.path.exists(path):
            raise RuntimeError(f"Exists: {self.path}")
        return Experimenter(
            data, path, data_names, sp=sp, sp_v=sp_v, splitter_params=splitter_params, title=title, data_key=data_key,
            logger = logger)

    def get_n_splits(self):
        return len(self.train_idx_list)

    def get_n_splits_inner(self):
        return len(self.train_idx_list[0])

    def add_metric(self, name, target_vars, output_var, metric_func, include_train=False):
        """Metric 인스턴스를 생성하여 추가

        Args:
            name: metric 이름
            target_vars: 타겟 변수 리스트 [(node_name, var), ...]
            output_var: 출력 변수
            metric_func: metric 함수
            include_train: train 결과 포함 여부 (기본값: False)

        Returns:
            Metric: 생성된 Metric 인스턴스
        """
        self._check_open()
        # __metric 폴더 생성 (최초 추가 시)
        metric_dir = self.path / "__metric"
        if not metric_dir.exists():
            metric_dir.mkdir(parents=True, exist_ok=True)

        metric = Metric(
            name=name,
            experimenter=self,
            target_vars=target_vars,
            output_var=output_var,
            metric_func=metric_func,
            include_train=include_train
        )
        self.metric[name] = metric
        self._save()
        return metric

    def add_stacking(self, name, target_vars, output_var, method='mean', include_target=True):
        """Stacking 인스턴스를 생성하여 추가

        Args:
            name: stacking 이름
            target_vars: 타겟 변수 리스트 [(node_name, var), ...]
            output_var: 출력 변수
            method: 집계 방법 (기본값: 'mean')
            include_target: 타겟 포함 여부 (기본값: True)

        Returns:
            Stacking: 생성된 Stacking 인스턴스
        """
        self._check_open()

        stacking = Stacking(
            experimenter=self,
            target_vars=target_vars,
            output_var=output_var,
            method=method,
            include_target=include_target
        )
        stacking.name = name
        stacking.save_config()
        self.stacking[name] = stacking
        self._save()
        return stacking

    def _validate_name(self, name):
        """Node 또는 NodeGroup 이름 검증

        Args:
            name: 검증할 이름

        Raises:
            ValueError: 이름이 유효하지 않을 경우
        """
        if name is None:
            return

        # '__' 포함 금지
        if '__' in name:
            raise ValueError(f"Name '{name}' cannot contain '__'")

        # 파일/폴더명으로 사용 불가한 문자 금지
        invalid_chars = ['/', '\\', '\0', '<', '>', ':', '"', '|', '?', '*']
        for char in invalid_chars:
            if char in name:
                raise ValueError(f"Name '{name}' cannot contain '{char}'")

    def get_grp_path(self, grp):
        if self.path is None:
            return None
        if isinstance(grp, str):
            grp = self.pipeline.get_grp(grp)
        path_parts = [grp.name]
        current = self.pipeline.get_grp(grp.parent)
        while current is not None:
            path_parts.insert(0, current.name)
            current = self.pipeline.get_grp(current.parent)
        return self.path / '/'.join(path_parts)

    def get_node_path(self, node):
        if isinstance(node, str):
            node = self.pipeline.get_node(node)
        grp_path = self.get_grp_path(node.grp)
        return grp_path / node.name

    def set_grp(self, name, role=None, processor=None, edges=None, method=None, parent=None, adapter=None, params=None, replace = False):
        self._check_open()
        result_obj = self.pipeline.set_grp(
            name, role, processor, edges, method, parent, adapter, params, replace
        )
        
        node_to_initialize = result_obj['affected_nodes']
        for node in node_to_initialize:
            node.initialize()

        for v in self.metric.values():
            v.reset_nodes(node_to_initialize)
        
        for v in self.stacking.values():
            v.reset_nodes(node_to_initialize)

        new_grp_path = self.get_grp_path(result_obj['obj'])

        if "old_grp" in result_obj:
            old_grp_path = self.get_grp_path(result_obj['old_obj'])
            if old_grp_path != new_grp_path:
                os.makedirs(new_grp_path, exist_ok=True)
                for fname in os.listdir(old_grp_path):
                    src_path = os.path.join(old_grp_path, fname)
                    dst_path = os.path.join(new_grp_path, fname)
                    shutil.move(src_path, dst_path)

            self.logger.info(f"Group '{name}' updated, {len(node_to_initialize)} node(s) affected")
        self._save()
        return result_obj

    def rename_grp(self, name_from, name_to):
        self._check_open()
        result_obj = self.pipeline.rename_grp(
            name_from, name_to
        )
        new_grp_path = self.get_grp_path(grp)
        os.makedirs(new_grp_path, exist_ok=True)
        for fname in os.listdir(old_grp_path):
            src_path = os.path.join(old_grp_path, fname)
            dst_path = os.path.join(new_grp_path, fname)
            shutil.move(src_path, dst_path)
        shutil.rmtree(old_grp_path)
        self._save()
    
    def remove_grp(self, name):
        self._check_open()
        self.pipeline.remove_grp(name)

        self.logger.info(f"Group '{name}' removed")
        self._save()


    def remove_node(self, name):
        """노드를 제거

        Args:
            name: 제거할 노드 이름

        Raises:
            ValueError: 노드가 존재하지 않거나, 자식 노드가 있는 경우
        """
        self._check_open()
        self.pipeline.remove_node(name)
        for v in self.metric.values():
            v.reset_nodes([name])
        
        for v in self.stacking.values():
            v.reset_nodes([name])
        
        self.logger.info(f"Node '{name}' removed")
        self._save()

    def finalize(self, nodes):
        self._check_open()
        if nodes is None:
            # 기존 동작: 모든 root group의 노드
            node_names = list(self.nodes.keys())
        elif isinstance(nodes, list):
            node_names = [n for n in nodes if n in self.nodes]
        elif isinstance(nodes, str):
            pat = re.compile(nodes)
            node_names = [k for k in self.nodes.keys() if k is not None and pat.search(k)]
        else:
            raise ValueError(f"nodes must be None, list, or str, got {type(nodes)}")
        target_nodes = list()
        for i in node_names:
            node = self.nodes[i]
            if type(node) != RootNode and node.grp.role == 'head' and node.status == 'built':
                self.logger.info(f"Finalize '{i}'")
                node.finalize()

    def reinitialize(self, nodes):
        self._check_open()
        if nodes is None:
            # 기존 동작: 모든 root group의 노드
            node_names = list(self.nodes.keys())
        elif isinstance(nodes, list):
            node_names = [n for n in nodes if n in self.nodes]
        elif isinstance(nodes, str):
            pat = re.compile(nodes)
            node_names = [k for k in self.nodes.keys() if k is not None and pat.search(k)]
        else:
            raise ValueError(f"nodes must be None, list, or str, got {type(nodes)}")
        target_nodes = list()
        for i in node_names:
            node = self.nodes[i]
            if type(node) != RootNode and node.status == 'finalized':
                self.logger.info(f"reinitialize '{i}'")
                node.initialize()

    def close_exp(self):
        if self.status != "open":
            raise RuntimeError("")
        for k, node_obj in self.node_objs.items():
            if node_obj.status == 'built':
                self.logger.info(f"Finalize '{k}'")
                node_obj.finalize()
        self.status = "closed"
    
    def reopen_exp(self):
        if self.status != "closed":
            raise RuntimeError("")
        for k, node in self.nodes.items():
            if type(node) != RootNode and node.grp.role == 'stage':
                self.logger.info(f"Intialize '{k}'")
                node.initialize()
        self.build()

    def set_node(
        self, name, grp, processor = None, edges = None,
        method = None, adapter = None, params = None, replace = False
    ):
        self._check_open()
        result_obj = self.pipeline.set_node(
            name, grp, processor, edges, method, adapter, params, replace
        )

        # 기존 노드를 업데이트한 경우, 하위 노드들도 재빌드
        if len(result_obj['affected_nodes']) > 0:
            affected_nodes = result_obj['affected_nodes']
            if descendants:
                self.logger.info(f"Affected {len(affected_nodes)} dependent node(s): {sorted(affected_nodes)}")
                for i in affected_nodes:
                    self.nodes[i].initialize()

                for v in self.metric.values():
                    v.reset_nodes(affected_nodes)
                
                for v in self.stacking.values():
                    v.reset_nodes(affected_nodes)
        
        self._save()
        return result_obj

    def build(self, nodes = None, rebuild = False):
        self._check_open()
        node_names = self.pipeline.get_node_names(nodes)
        target_nodes = list()
        for i in self.pipeline._get_effected_nodes([None]):
            node = self.pipeline.get_node(i)
            grp = self.pipeline.get_grp(node.grp)
            if grp.role == 'stage' and i not in self.node_objs:
                target_nodes.append(i)

        self.logger.info(f"Building {len(target_nodes)} node(s)")
        for node in target_nodes:
            if node in self.node_objs:
                node_obj = self.node_objs[node]
            else:
                node_obj = StageObj(self.get_node_path(node))
                self.node_objs[node] = node_obj
            node_obj.start_build()
        
        n_splits = self.get_n_splits()
        self.logger.start_progress("Build", n_splits)
        try:
            for i in range(n_splits):
                self.logger.update_progress(i)
                self.logger.start_progress("Node", len(target_nodes))
                for ni, node in enumerate(target_nodes):
                    self.logger.update_progress(ni)
                    self.logger._progress[-1][0] = node
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        node_obj = self.node_objs[node]
                        node_attrs = self.pipeline.get_node_attrs(node)
                        result_iter = node_obj.build_idx(
                            i, node_attrs, self.get_data(i, node_attrs['edges']), self.logger
                        )
                        for w in caught:
                            self.logger.warning(f"[{node}] fold {i}: {w.category.__name__}: {w.message}")
                self.logger.end_progress(len(target_nodes))
            self.logger.end_progress(n_splits)
        except Exception as e:
            self.logger.clear_progress()
            self.logger.info(f"Build failed at fold {i}, node '{node}': {type(e).__name__}: {e}")
            self.logger.info(traceback.format_exc())
            raise
        for node in target_nodes:
            node_obj = self.node_objs[node]
            node_obj.end_build()
        self.logger.info(f"Build complete: {len(target_nodes)} node(s)")
    
    def exp(self, nodes = None):
        self._check_open()
        node_names = self.pipeline.get_node_names(nodes)
        target_nodes = list()
        for i in self.pipeline._get_effected_nodes([None]):
            node = self.pipeline.get_node(i)
            grp = self.pipeline.get_grp(node.grp)
            if grp.role == 'head' and i not in self.node_objs:
                target_nodes.append(i)
        
        self.logger.info(f"Experimenting {len(target_nodes)} node(s)")

        # start_experiment for all nodes
        for node in target_nodes:
            if node in self.node_objs:
                node_obj = self.node_objs[node]
            else:
                node_obj = HeadObj(self.get_node_path(node))
                self.node_objs[node] = node_obj
            node_obj.start_exp()

        # _start for metrics and stackings
        for v in self.metric.values():
            for node in target_nodes:
                v._start(node)
        for v in self.stacking.values():
            for node in target_nodes:
                v._start(node)

        # experiment loop
        n_splits = self.get_n_splits()
        self.logger.start_progress("Exp", n_splits)
        try:
            for i in range(n_splits):
                self.logger.update_progress(i)
                # prepare target metrics data
                target_metrics = {
                    k: v._get_data(i) for k, v in self.metric.items()
                }

                self.logger.start_progress("Node", len(target_nodes))
                for ni, node in enumerate(target_nodes):
                    self.logger.update_progress(ni)
                    self.logger._progress[-1][0] = node
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        node_obj = self.node_objs[node]
                        node_attrs = self.pipeline.get_node_attrs(node)
                        result_iter = node_obj.exp_idx(
                            i, node_attrs, self.get_data(i, node_attrs['edges']), self.logger
                        )

                        stacks = {k: list() for k in self.stacking.keys()}
                        sub_metrics = {k: list() for k in self.metric.keys()}

                        for n, result_data in enumerate(result_iter):
                            # collect metrics
                            for k, v in self.metric.items():
                                sub_metric = v._get_metric(target_metrics[k][n], result_data)
                                sub_metric = {k_sub: v_sub for k_sub, v_sub in sub_metric.items()}
                                sub_metrics[k].append(sub_metric)
                            # collect stacking data
                            for k, v in self.stacking.items():
                                _valid = v._get_valid(result_data)
                                if _valid is not None:
                                    stacks[k].append(_valid)

                        for w in caught:
                            self.logger.warning(f"[{node}] fold {i}: {w.category.__name__}: {w.message}")

                    # set metrics
                    for k, v in self.metric.items():
                        v._set_metric(node, i, sub_metrics[k])
                    # aggregate and stack
                    for k, v in self.stacking.items():
                        if len(stacks[k]) > 0:
                            stk = v._aggregate(iter(stacks[k]))
                            v._stack(node, i, stk)
                self.logger.end_progress(len(target_nodes))
            self.logger.end_progress(n_splits)
        except Exception as e:
            self.logger.clear_progress()
            self.logger.info(f"Exp failed at fold {i}, node '{node}': {type(e).__name__}: {e}")
            self.logger.info(traceback.format_exc())
            # _start for metrics and stackings
            for v in self.metric.values():
                v.reset_nodes(target_nodes)
            for v in self.stacking.values():
                v.reset_nodes(target_nodes)
            raise

        # end_experiment for all nodes
        for node in target_nodes:
            node_obj = self.node_objs[node]
            node_obj.end_exp()

        # _end for metrics and stackings
        for v in self.metric.values():
            for node in target_nodes:
                v._end(node)
        for v in self.stacking.values():
            for node in target_nodes:
                v._end(node)

        self.logger.info(f"Experimentation complete: {len(target_nodes)} node(s)")

    def get_data(self, idx, edges):
        def ret_data_func(data_dict):
            result = {}
            for _ in range(self.get_n_splits_inner()):
                for k, it in data_dict.items():
                    z = next(it)
                    train_sub, valid_sub, outer_valid_sub = list(), list(), list()
                    for (train_data, train_v_data), outer_valid_data in z:
                        train_sub.append(train_data)
                        if train_v_data is not None:
                            valid_sub.append(train_v_data)
                        outer_valid_sub.append(outer_valid_data)

                    train_concat = type(train_sub[0]).concat(train_sub, axis=1)
                    outer_concat = type(outer_valid_sub[0]).concat(outer_valid_sub, axis=1)
                    if len(valid_sub) > 0:
                        valid_concat = type(valid_sub[0]).concat(valid_sub, axis=1)
                        result[k] = ((train_concat, valid_concat), outer_concat)
                    else:
                        result[k] = ((train_concat, None), outer_concat)
                yield result

        data_dict = {}
        for key, edge_list in edges.items():
            key_data_list = []
            for node_name, var in edge_list:
                key_data_list.append(self.get_node_output(idx, node_name, var))
            data_dict[key] = zip(*key_data_list)
        return ret_data_func(data_dict)
    
    def get_data_train(self, idx, edges):
        def ret_data_func(data_dict):
            result = {}
            for _ in range(self.get_n_splits_inner()):
                for k, it in data_dict.items():
                    z = next(it)
                    train_sub, valid_sub = list(), list()
                    for train_data, train_v_data in z:
                        train_sub.append(train_data)
                        if train_v_data is not None:
                            valid_sub.append(train_v_data)
                    train_concat = type(train_sub[0]).concat(train_sub, axis=1)
                    if len(valid_sub) > 0:
                        valid_concat = type(valid_sub[0]).concat(valid_sub, axis=1)
                        result[k] = (train_concat, valid_concat)
                    else:
                        result[k] = (train_concat, None)
                yield result

        data_dict = {}
        for key, edge_list in edges.items():
            key_data_list = []
            for node_name, var in edge_list:
                key_data_list.append(self.get_node_train_output(idx, node_name, var))
            data_dict[key] = zip(*key_data_list)
        return ret_data_func(data_dict)
    
    def get_data_valid(self, idx, edges):
        """외부 검증 데이터에 대한 처리 결과를 가져옴

        Args:
            idx: outer fold 인덱스
            edges: {key: [(node_name, var), ...], ...} 형태의 edge dict

        Yields:
            dict: {key: valid_concat, ...} 각 key별로 concat된 외부 검증 데이터 결과
        """
        def ret_data_func(data_dict):
            result = {}
            for _ in range(self.get_n_splits_inner()):
                for k, it in data_dict.items():
                    z = next(it)
                    outer_valid_sub = list()
                    for outer_valid_data in z:
                        outer_valid_sub.append(outer_valid_data)
                    outer_concat = type(outer_valid_sub[0]).concat(outer_valid_sub, axis=1)
                    result[k] = outer_concat
                yield result

        data_dict = {}
        for key, edge_list in edges.items():
            key_data_list = []
            for node_name, var in edge_list:
                key_data_list.append(self.get_node_valid_output(idx, node_name, var))
            data_dict[key] = zip(*key_data_list)
        return ret_data_func(data_dict)

    def split(self, edges):
        for idx in range(len(self.train_idx_list)):
            yield self.get_data(idx, edges)
    
    def get_node_output(self, idx, node, v = None):
        if node is None:
            outer_valid_data = self.data.iloc(self.valid_idx_list[idx])
            if v is not None:
                outer_valid_data = outer_valid_data.select_columns(v)
            def ret_func():
                for train_v_idx, valid_v_idx in self.train_idx_list[idx]:
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
        it = self.cache.get_data(node, "all", idx, v)
        if it is not None:
            return it
        node_attrs = self.pipeline.get_node_attrs(node)
        it = self.get_data(idx, node_attrs['edges'])
        sub = self.node_objs[node].get_objs(idx)
        if True:
            cache_data = list()
        else:
            cache_data = None
        def ret_func():
            for data_dict, (obj, train_, info) in zip(it, sub):
                # X key로 입력 데이터 가져오기
                (train_X, train_v_X), valid_X = data_dict['X']

                # train data 처리
                if train_ is None:
                    train_result = obj.process(train_X)
                else:
                    train_result = train_

                # train_v data 처리
                if train_v_X is not None:
                    train_v_result = obj.process(train_v_X)
                else:
                    train_v_result = None

                # valid data 처리 (외부 fold의 valid)
                valid_result = obj.process(valid_X)

                # 필요하면 컬럼 필터링
                if v is not None:
                    X = resolve_columns(train_result, v, processor=obj)
                    train_result = train_result.select_columns(X)
                    if train_v_result is not None:
                        train_v_result = train_v_result.select_columns(X)
                    valid_result = valid_result.select_columns(X)

                yld = (train_result, train_v_result), valid_result
                if cache_data is not None:
                    cache_data.append(yld)
                yield yld
        if True:
            self.cache.put_data(node, "all", idx, v, cache_data)
        return ret_func()

    def get_node_train_output(self, idx, node, v=None):
        if node is None:
            def ret_func():
                for train_v_idx, valid_v_idx in self.train_idx_list[idx]:
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
        it = self.cache.get_data(node, "train", idx, v)
        if it is not None:
            return it
        node_attrs = self.pipeline.get_node_attrs(node)
        it = self.get_data_valid(idx, node_attrs['edges'])
        sub = self.node_objs[node].get_objs(idx)
        def ret_func():
            for data_dict, (obj, train_, info) in zip(it, sub):
                # X key로 입력 데이터 가져오기
                train_X, train_v_X = data_dict[self.node.X]
                train_result = obj.process(train_X) if train_ is None else train_
                if train_v_X is not None:
                    train_v_result = obj.process(train_v_X)
                else:
                    train_v_result = None
                # 필요하면 컬럼 필터링
                if v is not None:
                    X = resolve_columns(train_result, v, processor=obj)
                    train_result = train_result.select_columns(X)
                    if train_v_result is not None:
                        train_v_result = train_v_result.select_columns(X)
                if self.cache_t is not None:
                    self.cache_t.append((train_result, train_v_result))
                yld = train_result, train_v_result
                if cache_data is not None:
                    cache_data.append(yld)
                yield yld
        if True:
            self.cache.put_data(node, "train", idx, v, cache_data)
        return ret_func()
    
    def get_node_valid_output(self, idx, node, v=None):
        if node is None:
            outer_valid_data = self.data.iloc(self.valid_idx_list[idx])
            if v is not None:
                outer_valid_data = outer_valid_data.select_columns(v)

            def ret_func():
                for _ in range(self.get_n_splits_inner()):
                    yield outer_valid_data

            return ret_func()
        it = self.cache.get_data(node, "valid", idx, v)
        if it is not None:
            return it
        node_attrs = self.pipeline.get_node_attrs(node)
        it = self.get_data_valid(idx, node_attrs['edges'])
        sub = self.node_objs[node].get_objs(idx)
        if True:
            cache_data = list()
        else:
            cache_data = None
        def ret_func():
            for data_dict, (obj, train_, info) in zip(it, sub):
                # X key로 valid 데이터 가져오기
                valid_X = data_dict['X']
                # valid data 처리 (외부 fold의 valid)
                valid_result = obj.process(valid_X)

                # 필요하면 컬럼 필터링
                if v is not None:
                    X = resolve_columns(valid_result, v, processor=obj)
                    valid_result = valid_result.select_columns(X)
                yld = valid_result
                if cache_data is not None:
                    cache_data.append(yld)
                yield yld

        if True:
            self.cache.put_data(node, "valid", idx, v, cache_data)
        return ret_func()
    
    def get_node_info(self):
        lines = [f"# Experiment Pipeline Summary\n"]
        lines.append(f"- **Root**: {type(self.root).__name__}\n")

        for name, node in self.nodes.items():
            if name is None:
                continue
            processor_name = node.processor.__name__
            edges_info_parts = []
            for key, edge_list in node.edges.items():
                edge_strs = [f"{n or 'Root'}{f'[{v}]' if v else ''}" for n, v in edge_list]
                edges_info_parts.append(f"{key}: [{', '.join(edge_strs)}]")
            edges_info = ", ".join(edges_info_parts)
            lines.append(f"## {name}")
            lines.append(f"- **Processor**: {processor_name}")
            lines.append(f"- **Method**: {node.method}")
            lines.append(f"- **Edges**: {edges_info}")

            descendants = self._find_descendants(name)
            if descendants:
                lines.append(f"- **Descendants**: {sorted(descendants)}")
            lines.append("")

        return "\n".join(lines)

    def desc_spec(self):
        """실험 스펙을 Markdown으로 반환"""
        return desc_spec(self)

    def desc_node_vars(self, node_name, idx):
        """특정 노드의 입력/출력 변수를 분석

        Args:
            node_name: 대상 노드 이름
            idx: 외부 fold 인덱스

        Returns:
            list: [(입력변수 리스트, 출력변수 리스트, 해당 내부 폴드 index 리스트), ...]
                  등장 빈도의 내림차순으로 정렬
        """
        return desc_node_vars(self, node_name, idx)

    def get_objs(self, node_name, idx):
        if node_name not in self.nodes or node_name is None:
            raise ValueError(f"Node '{node_name}' not found")

        node = self.nodes[node_name]

        # 노드가 빌드되지 않았으면 에러
        if node.status != 'built':
            raise ValueError(f"Node '{node_name}' status should be built")

        # 외부 fold의 내부 fold들: [(processor, train_v, info), ...]
        return node.get_objs(idx)
    
    def get_node_vars(self, node_name, idx):
        """특정 노드의 입력/출력 변수를 가져옴

        Args:
            node_name: 대상 노드 이름
            idx: 외부 fold 인덱스

        Returns:
            list: [(입력변수 리스트, 출력변수 리스트, 해당 내부 폴드 index 리스트), ...]
                  등장 빈도의 내림차순으로 정렬
        """
        if node_name not in self.nodes or node_name is None:
            raise ValueError(f"Node '{node_name}' not found")

        node = self.nodes[node_name]

        # 노드가 빌드되지 않았으면 에러
        if node.status != 'built':
            raise ValueError(f"Node '{node_name}' status should be built")

        # 외부 fold의 내부 fold들: [(processor, train_v, info), ...]
        inner_folds = node.get_objs(idx)

        # (입력변수 튜플, 출력변수 튜플) -> 내부 fold index 리스트
        var_map = {}

        for inner_idx, (processor, train_v, info) in enumerate(inner_folds):
            # 입력 변수와 출력 변수 가져오기
            input_vars = tuple(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else ()
            output_vars = tuple(processor.output_vars) if hasattr(processor, 'output_vars') and processor.output_vars is not None else ()

            # 튜플 키 생성
            key = (input_vars, output_vars)

            if key not in var_map:
                var_map[key] = []
            var_map[key].append(inner_idx)

        # 결과 리스트 생성: [(입력변수 리스트, 출력변수 리스트, 내부 폴드 index 리스트), ...]
        result = []
        for (input_vars, output_vars), fold_indices in var_map.items():
            result.append((list(input_vars), list(output_vars), fold_indices))

        # 등장 빈도(내부 폴드 개수)의 내림차순으로 정렬
        result.sort(key=lambda x: len(x[2]), reverse=True)

        return result

    def get_edges_var(self, edges):
        class _ColHolder:
            def __init__(self, columns):
                self._columns = columns
            def get_columns(self):
                return self._columns

        var_map = {}

        for idx in range(self.get_n_splits()):
            n_inner = len(self.train_idx_list[idx])
            # edges는 dict: {key: [(node_name, var), ...], ...}
            edge_objs = {}
            for key, edge_list in edges.items():
                edge_objs[key] = []
                for node_name, var in edge_list:
                    if node_name is None:
                        edge_objs[key].append((None, var, None))
                    else:
                        node_obj = self.node_objs[node_name]
                        edge_objs[key].append((node_name, var, node_obj.get_objs(idx)))

            for inner_idx in range(n_inner):
                collected = {}
                for key, objs_list in edge_objs.items():
                    collected[key] = []
                    for node_name, var, objs in objs_list:
                        obj = next(objs)
                        if node_name is None:
                            cols = self.data.get_columns()
                            proc = None
                        else:
                            proc = obj[0]
                            cols = list(proc.output_vars) if proc.output_vars is not None else []

                        if var is not None:
                            cols = resolve_columns(_ColHolder(cols), var, processor=proc)

                        collected[key].extend(cols)

                # key를 정렬된 형태로 튜플화
                key_tuple = tuple((k, tuple(v)) for k, v in sorted(collected.items()))
                if key_tuple not in var_map:
                    var_map[key_tuple] = []
                var_map[key_tuple].append((idx, inner_idx))

        result = []
        for vars_tuple, fold_indices in var_map.items():
            # dict로 복원
            vars_dict = {k: list(v) for k, v in vars_tuple}
            result.append((vars_dict, fold_indices))

        result.sort(key=lambda x: len(x[1]), reverse=True)

        return result

    def _get_grp_load_order(self):
        """NodeGroup 로딩 순서를 BFS로 계산 (parent가 없는 그룹부터 시작)

        Returns:
            list: 로딩 순서에 맞는 (grp_name, parent_grp_name) 튜플 리스트
        """
        result = []
        queue = [(grp.name, None) for grp in self.grps.values() if grp.parent is None]

        while queue:
            grp_name, parent_name = queue.pop(0)
            result.append((grp_name, parent_name))
            grp = self.grps[grp_name]
            for child_grp in grp.child_grps:
                queue.append((child_grp.name, grp_name))

        return result

    def _get_node_load_order(self):
        """Node 로딩 순서를 계산 (_get_effected_nodes 사용)

        Returns:
            list: 로딩 순서에 맞는 (node_name, grp_name) 튜플 리스트
        """
        effected = self.pipeline._get_effected_nodes([None])
        result = []
        for node in effected:
            result.append((node.name, node.grp.name))
        return result

    def _save(self, filepath=None):
        """Experimenter 객체를 파일로 저장

        Args:
            filepath: 저장할 파일 경로 (None이면 self.path / '__exp.pkl')
        """
        return
        if filepath is None:
            filepath = self.path / '__exp.pkl'

        # 저장할 데이터 구성 (data는 저장하지 않음)
        save_data = {
            'data_key': self.data_key,
            'title': self.title,
            'sp': self.sp,
            'sp_v': self.sp_v,
            'splitter_params': self.splitter_params,
            'exp_id': self.exp_id,
            'grp_load_order': self._get_grp_load_order(),
            'node_load_order': self._get_node_load_order(),
            'metric_keys': list(self.metric.keys()),
            'stacking_keys': list(self.stacking.keys()),
            "status": self.status
        }

        # print(f"💾 Saving Experimenter to {filepath}...")
        with open(filepath, 'wb') as f:
            pkl.dump(save_data, f)
        """
        print(f"✅ Experimenter saved successfully")
        print(f"   - {len(save_data['node_load_order'])} node(s)")
        print(f"   - {len(save_data['grp_load_order'])} group(s)")
        print(f"   - {len(save_data['metric_keys'])} metric(s)")
        print(f"   - {len(save_data['stacking_keys'])} stacking(s)")
        """

    @staticmethod
    def load(filepath, data, data_key=None):
        """파일에서 Experimenter 객체를 불러옴

        Args:
            filepath: 불러올 파일 경로
            data: 실험에 사용할 데이터
            data_key: 데이터 식별자 (저장된 data_key와 비교하여 검증)

        Returns:
            Experimenter: 불러온 Experimenter 객체

        Raises:
            ValueError: 저장된 data_key와 전달된 data_key가 일치하지 않는 경우
        """
        from ._metric import Metric
        from ._stacking import Stacking

        filepath = Path(filepath)
        with open(filepath / '__exp.pkl', 'rb') as f:
            save_data = pkl.load(f)

        # data_key 검증 (저장된 data_key가 None이 아닌 경우에만)
        saved_data_key = save_data.get('data_key')
        if saved_data_key is not None and saved_data_key != data_key:
            raise ValueError(
                f"data_key mismatch: saved='{saved_data_key}', provided='{data_key}'"
            )

        # Experimenter 생성자 활용
        exp = Experimenter(
            data=data,
            path=filepath,
            sp=save_data['sp'],
            sp_v=save_data['sp_v'],
            splitter_params=save_data['splitter_params'],
            title=save_data['title'],
            data_key=saved_data_key
        )
        # exp_id를 저장된 값으로 복원
        exp.exp_id = save_data['exp_id']

        # NodeGroup 복원 (로딩 순서대로)
        for grp_name, parent_name in save_data['grp_load_order']:
            parent = exp.grps.get(parent_name) if parent_name else None
            grp = exp.load_grp(grp_name, parent)
            exp.grps[grp_name] = grp

        # Node 복원 (로딩 순서대로)
        for node_name, grp_name in save_data['node_load_order']:
            grp = exp.grps[grp_name]
            node = Node.load(exp, grp, node_name)
            exp.nodes[node_name] = node

        # output_edges 재구성
        for node_name, node in exp.nodes.items():
            if node_name is None:
                continue
            for key, edge_list in node.edges.items():
                for edge_name, _ in edge_list:
                    if edge_name in exp.nodes:
                        parent_node = exp.nodes[edge_name]
                        if node_name not in parent_node.output_edges:
                            parent_node.output_edges.append(node_name)

        # Metric 복원
        for metric_name in save_data['metric_keys']:
            metric = Metric.load_from_file(exp, metric_name)
            exp.metric[metric_name] = metric

        # Stacking 복원
        for stacking_name in save_data['stacking_keys']:
            stacking = Stacking.load_from_file(exp, stacking_name)
            exp.stacking[stacking_name] = stacking

        exp.logger.info(f"Loaded: {len(exp.nodes) - 1} node(s), {len(exp.grps)} group(s), {len(exp.train_idx_list)} fold(s)")
        exp.status = save_data['status']
        return exp

    def get_result(self, node, idx, result, params = {}):
        node_attrs = self.pipeline.get_node_attrs(node)
        adapter = node_attrs['adapter']
        if result not in adapter.result_objs:
            raise ValueError(f"{result} Unsupported result")
        result_func = adapter.result_objs[result][0]

        for i in self.node_objs[node].get_objs(idx):
            yield result_func(i[0], **params)

    def get_results(self, node, result, params = {}):
        for i in range(self.get_n_splits()):
            yield list(
                self.get_result(node, i, result, params)
            )

    def get_results_agg(self, node, result, params = {}, agg_inner = True, agg_outer = True):
        if agg_outer and not agg_inner:
            raise ValueError("agg_outer requires agg_inner to be True")
        node_attrs = self.pipeline.get_node_attrs(node)
        adapter = node_attrs['adapter']
        if not adapter.result_objs[result][1]:
            raise ValueError(f"Result '{result}' is not mergeable across folds")
        l = list()
        for no, i in enumerate(self.get_results(node, result, params)):
            l.append(pd.concat([j.rename(no_i) for no_i, j in enumerate(i)], axis = 1).stack().rename(no))
        df = pd.concat(l, axis=1)
        if agg_inner:
            df = df.groupby(level=[i for i in range(len(df.index.levels) - 1)]).mean()
            if agg_outer:
                return df.mean(axis=1)
        return df

def create_like(exp, data, path, data_names=None, sp=None, sp_v=None, splitter_params=None, title=None, data_key=None):
    """기존 Experimenter의 구조를 복제하여 새로운 Experimenter 생성

    Args:
        exp: 구조를 복제할 원본 Experimenter
        data: 새로운 데이터
        path: 작업 디렉토리 경로
        data_names: 새로운 데이터의 컬럼명 (None이면 자동)
        sp: 외부 splitter (None이면 원본과 동일)
        sp_v: 내부 splitter (None이면 원본과 동일, "remove"이면 제거)
        splitter_params: splitter에 전달할 파라미터 (None이면 원본과 동일)
        title: 실험 타이틀 (None이면 원본과 동일)
        data_key: 데이터 식별자 (None이면 원본과 동일)

    Returns:
        Experimenter: 새로 생성된 Experimenter 인스턴스
    """
    exp.logger.info("Creating new Experimenter with same structure...")

    if sp is None:
        sp = exp.sp
    if sp_v is None:
        sp_v = exp.sp_v
    elif sp_v == "remove":
        sp_v = None
    if splitter_params is None:
        splitter_params = exp.splitter_params.copy() if exp.splitter_params else None
    if title is None:
        title = exp.title
    if data_key is None:
        data_key = exp.data_key

    # 새 Experimenter 생성
    new_exp = Experimenter(
        data,
        path=path,
        data_names=data_names,
        sp=sp,
        sp_v=sp_v,
        splitter_params=splitter_params,
        title=title,
        data_key=data_key
    )
    new_exp.logger.info(f"Created base Experimenter with {len(new_exp.train_idx_list)} fold(s)")

    # 그룹 복제 (부모-자식 관계를 유지하기 위해 위상 정렬)
    # 1. 최상위 그룹부터 BFS로 복제
    grp_mapping = {}  # 원본 그룹명 -> 새 그룹 객체

    # 최상위 그룹 찾기 (parent가 None인 그룹)
    top_level_grps = [grp for grp in exp.grps.values() if grp.parent is None]

    def clone_group_recursive(orig_grp, parent_name=None):
        """그룹을 재귀적으로 복제"""
        # edges dict 복사
        edges_copy = {k: list(v) for k, v in orig_grp.edges.items()}
        new_grp = new_exp.set_grp(
            name=orig_grp.name,
            role=orig_grp.role,
            processor=orig_grp.processor,
            edges=edges_copy,
            X=orig_grp.X,
            y=orig_grp.y,
            method=orig_grp.method,
            parent=parent_name,
            adapter=orig_grp.adapter,
            params=orig_grp.params.copy() if orig_grp.params else {}
        )
        grp_mapping[orig_grp.name] = new_grp

        # 자식 그룹들도 복제
        for child_grp in orig_grp.child_grps:
            clone_group_recursive(child_grp, parent_name=orig_grp.name)

    # 최상위 그룹부터 재귀적으로 복제
    for grp in top_level_grps:
        clone_group_recursive(grp)

    new_exp.logger.info(f"Cloned {len(exp.grps)} group(s)")

    # 노드 복제 (위상 정렬: Root부터 BFS)
    # 1. 노드의 우선순위 계산 (BFS)
    node_priorities = {}
    queue = [(None, 0)]  # Root부터 시작

    while queue:
        current_node, priority = queue.pop(0)

        if current_node in node_priorities:
            continue

        node_priorities[current_node] = priority

        # current_node를 edge로 참조하는 child 노드들 찾기
        for name in current_node.output_edges :
            if name is not None and name not in node_priorities:
                queue.append((name, priority + 1))
                break

    # 우선순위 순으로 노드 정렬 (Root 제외)
    sorted_nodes = sorted(
        [(name, node) for name, node in exp.nodes.items() if name is not None],
        key=lambda x: node_priorities.get(x[0], float('inf'))
    )

    # 노드 복제
    for name, orig_node in sorted_nodes:
        if orig_node.org_attr is not None:
            org = orig_node.org_attr
            # edges dict 복사
            edges_copy = {k: list(v) for k, v in org['edges'].items()} if org['edges'] else {}
            new_exp.set_node(
                name,
                grp=orig_node.grp.name,
                processor=org['processor'],
                edges=edges_copy,
                method=org['method'],
                adapter=org['adapter'],
                params=org['params'].copy() if org['params'] else {}
            )

    new_exp.logger.info(f"Structure cloning complete: {len(exp.grps)} group(s), {len(sorted_nodes)} node(s)")

    return new_exp