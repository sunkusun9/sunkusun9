import polars as pl
import pandas as pd
import numpy as np
import pickle as pkl

class Node():
    def __init__(self, experimenter, name, processor, edges, y = None, method = 'fit', split_train = False):
        self.name = name
        self.processor = processor
        self.method = method
        self.edges = edges
        self.experimenter = experimenter
        self.spllt_train = split_train
        self.y = None
        if method == 'fit':
            self._fit()
        else:
            self._fit_transform()

    def get_input(self):
        for i in edges:
            self.experimenter.get_node_value('name'):
    def _fit(self):
        

class Experimenter():
    def __init__(self, data, data_names = None, sp = ShuffleSplit(n_splits = 1, random_state=1), sp_v = None, **args):
        self.train_idx_list = list()
        self.valid_idx_list = list()
        self.root = data
        split_params = {}
        if type(data) in [pd.DataFrame]:
            if data_names is None:
                data_names = data.columns.tolist()
            for k, v in args.items():
                split_params[k] = self.data[v]
        else:
            data_names = np.arange(data.shape[-1])
            for k, v in args.items():
                split_params[k] = data[:, v]
       
        for train_idx, valid_idx in sp.split(data, **split_params):
            if sp_v is not None:
                if type(data) in [pd.DataFrame]:
                    split_params = {'X': data.iloc[train_idx]}
                    for k, v in args.items():
                        split_params[k] = split_params['X'][v]
                else:
                    split_params = {'X': data[train_idx]}
                    for k, v in args.items():
                        split_params[k] = split_params['X'][:, v]
                self.train_idx_list.append([
                    (train_idx[train_v_idx], train_idx[valid_v_idx])
                    for train_v_idx, valid_v_idx in sp_v.split(**split_params)
                ])
            else:
                self.train_idx_list.append([
                    (train_idx, None)
                ])
            self.valid_idx_list.append(valid_idx)
        self.nodes = dict()
            
    def set_node(self, name, edges, node, method = 'transform', split_train = False):

    def get_node_value(self, idx, values, name = None):
        if name is None:
            v = self.roote
        else:
            v = self.nodes[name].transform(idx)
        if type(v) in [pd.DataFrame]:
            return v.iloc[idx][values]
        else
            return v[idx, values]

    def get_node_inputs(self, idx, name):
        node = self.nodes[name]
        node.edges

        
    def split(self, edges, split_train = True):
        def idx_func(x):
            return [
                i.iloc[x] if type(i) in [pd.DataFrame, pd.Series] else i[x]
                for i in [self.data]
            ]
        if split_train:
            def train_splitter(train_idx_list):
                for train_idx, train_v_idx in train_idx_list:
                    if train_v_idx is None:
                        yield idx_func(train_idx), None
                    else:
                        yield idx_func(train_idx), idx_func(train_v_idx)
            for train_idx_list, valid_idx in zip(self.train_idx_list, self.valid_idx_list):
                yield train_splitter(train_idx_list), idx_func(valid_idx)
        else:
            def train_splitter(train_idx_list):
                for train_idx, train_v_idx in train_idx_list:
                    if train_v_idx is None:
                        yield idx_func(train_idx), None
                    else:
                        yield idx_func(np.hstack([train_idx, train_v_idx])), None
            for train_idx_list, valid_idx in zip(self.train_idx_list, self.valid_idx_list):
                yield train_splitter(train_idx_list), idx_func(valid_idx)