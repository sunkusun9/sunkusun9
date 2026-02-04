import uuid
import pickle as pkl
from .adapter import get_adapter
from ._node_processor import TransformProcessor, PredictProcessor, resolve_columns
import numpy as np
import pandas as pd
import os
import shutil
import time


def _build_sub(node, data_dict, fit_process):
    if node.method in ['transform', 'fit_transform']:
        obj = TransformProcessor(node.processor, adapter=node.adapter, params = node.params)
    else:
        obj = PredictProcessor(node.processor, method=node.method, adapter=node.adapter, params = node.params)

    start_time = time.time()
    if fit_process:
        result = obj.fit_process(data_dict, node.X, node.y)
    else:
        result = None
        obj.fit(data_dict, node.X, node.y)
    elapsed_time = time.time() - start_time

    # X key로 shape 정보 가져오기
    (train_X, train_v_X), _ = data_dict[node.X]
    info = {
        'build_id': str(uuid.uuid4()),
        'fit_time': elapsed_time,
        'train_shape': train_X.get_shape() if train_X is not None else None,
        'train_v_shape': train_v_X.get_shape() if train_v_X is not None else None
    }
    return obj, result, info

def _build_iter(train, node):
    if node.method in ['transform', 'predict', 'predict_proba']:
        fit_process = False
    elif node.method in ['fit_transform', 'fit_predict']:
        fit_process = True
    else:
        raise ValueError(f"Unknown processor_type: {self.node.method}")
    
    for data_dict in train:
        yield _build_sub(data_dict, fit_process)

class ExpObj():
    def __init__(self, experimenter, node):
        self.experimenter = experimenter
        self.node = node
        
    def get_result(self, idx, result, params={}):
        if result not in self.node.adapter.result_objs:
            raise ValueError(f"{result} Unsupported result")
        result_func = self.node.adapter.result_objs[result][0]
        for i in self.get_objs(idx):
            yield result_func(i[0], **params)

class StageObj():
    def __init__(self, use_cache=True):
        self.experimenter = experimenter
        self.use_cache = use_cache
        self._unload_cache()

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

    def start_build(self):
        self.initialize()
        self.objs_ = list()
        if not os.path.isdir(self.path):
            os.makedirs(self.path, exist_ok = True)
    
    def build_idx(self, train_data, idx):
        if idx != len(self.objs_):
            raise RuntimeError(f"{node.name}: Build sequence is not valid")

        # train_data = experimenter.get_data(idx, node.edges)
        objs = list()
        for no, obj in enumerate(_build_iter(train, node)):
            filename = experimenter.get_node_path(node) / ('obj' + str(idx) + '_'  + str(no) + '.pkl')
            with open(filename, 'wb') as f:
                pkl.dump(obj, f)
            objs.append(obj)
        self.objs_.append(objs)
    
    def get_objs(self, idx):
        if self.status != 'built':
            raise RuntimeError(f"Node '{self.node.name}' must be built before accessing objects (status='{self.status}')")
        
        for obj, train_, spec in self.objs_[idx]:
            yield obj, train_, spec
        
    def clear(self):
        path = self.path
        if os.path.isdir(path):
            shutil.rmtree(path)
        self.objs_ = None


class HeadObj():
    def __init__(self, experimenter, node, use_cache=True):
        self.experimenter = experimenter
        self.node = node
        
        self.use_cache = use_cache
        self.status = None
        self.save_info()

    def start_exp(self):
        if self.status == "finalized":
            raise RuntimeError(f"Node '{self.node.name}' is finalized and cannot be re-experimented")
        else:
            if not os.path.isdir(self.path):
                os.makedirs(self.path, exist_ok = True)

    def exp_idx(self, idx, include_output = True, finalize = False):
        grp_obj = self.experimenter.grps[self.node.grp]
        if grp_obj.role != 'head':
            raise RuntimeError(f"Node '{self.node.name}' is not a head node (role='{grp_obj.role}')")
        if self.status == "built":
            def _iter_func():
                with open(filename, 'rb') as f:
                    for no in self.experimenter.get_n_splits_inner():
                        filename = self.experimenter.get_node_path(self.node) / ('obj' + str(idx) + '_'  + str(no).pkl')
                        with open(filename, 'rb') as f:
                            obj, train_, spec = pkl.load(f)
                        yield obj
            objs_iter = _iter_func()
        elif self.status == "finalized":
            raise RuntimeError(f"Node '{self.node.name}' is finalized and cannot be re-experimented")
        else:
            train = self.experimenter.get_data(idx, self.node.edges)
            objs_iter = _build_iter(train, self.node)
        ret = list()
        it = self.experimenter.get_data(idx, self.node.edges)
        result_list = list()
        no = 0
        for data_dict, (obj, train_, spec) in zip(it, objs_iter):
            # X key로 데이터 가져오기
            (train_X, train_v_X), valid_X = data_dict[self.node.X]
            sub_result = {'spec': spec, 'object': obj}
            if include_output:
                if train_ is None:
                    train_result = obj.process(train_X)
                else:
                    train_result = train_
                if train_v_X is not None:
                    train_v_result = obj.process(train_v_X)
                else:
                    train_v_result = None
                sub_result['output_train'] = (train_result, train_v_result)
                sub_result['output_valid'] = obj.process(valid_X)
            yield sub_result
            if self.status is None and (not finalize):
                filename = self.experimenter.get_node_path(self.node) / ('obj' + str(idx) + '_'  + str(no).pkl')
                with open(filename, 'wb') as f:
                    pkl.dump((obj, train_, spec), f)
            no += 1
    
    def end_exp(self, finalize = False):
        if finalize:
            self.status = "finalized"
        else:
            self.status = "built"

    def initialize(self):
        if self.status is None:
            return
        path = self.path
        if os.path.isdir(path):
            shutil.rmtree(path)
        self.status = None
        self.objs_ = None

    def finalize(self):
        if self.status != 'built':
            raise RuntimeError(f"Node '{self.node.name}' must be built before finalize (status='{self.status}')")
        if hasattr(self, 'objs_') and self.objs_ is not None:
            del self.objs_
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
        self.status = "finalized"
