import re
import json
import pickle
import numpy as np
from pathlib import Path
from modeler._data_wrapper import DataWrapper
from ._node_processor import resolve_columns

class Stacking:
    def __init__(self, experimenter, target_edges, output_var, method='mean', include_target=True):
        self.experimenter = experimenter
        self.target_edges = target_edges
        self.output_var = output_var
        self.method = method
        self.include_target = include_target
        self.columns = {}
        self.stacking = {}  # 진행 중인 실험의 중간 결과만 담김
        self.name = None  # add_stacking에서 설정됨

        # self._build_target_value()
        # self._build_sort_order()

    @property
    def stacking_dir(self):
        """stacking 데이터를 저장할 디렉토리 경로"""
        if self.name is None:
            raise ValueError("Stacking name is not set. Call experimenter.add_stacking first.")
        return self.experimenter.path / "__stacking" / self.name

    def _ensure_stacking_dir(self):
        """stacking 디렉토리가 존재하는지 확인하고 없으면 생성"""
        stacking_dir = self.stacking_dir
        if not stacking_dir.exists():
            stacking_dir.mkdir(parents=True, exist_ok=True)
        return stacking_dir

    def save_config(self):
        """target_edges, output_var, method, include_target를 파일로 저장"""
        stacking_dir = self._ensure_stacking_dir()
        config = {
            'target_edges': self.target_edges,
            'output_var': self.output_var,
            'method': self.method,
            'include_target': self.include_target,
        }
        config_path = stacking_dir / "__config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)

    def load_config(self):
        """파일에서 target_edges, output_var, method, include_target를 로드"""
        config_path = self.stacking_dir / "__config.pkl"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        self.target_edges = config['target_edges']
        self.output_var = config['output_var']
        self.method = config['method']
        self.include_target = config['include_target']

    @classmethod
    def load_from_file(cls, experimenter, name):
        """저장된 Stacking 정보를 불러와서 Stacking 인스턴스 생성

        Args:
            experimenter: Experimenter 인스턴스
            name: stacking 이름

        Returns:
            Stacking: 복원된 Stacking 인스턴스
        """
        config_path = experimenter.path / "__stacking" / name / "__config.pkl"
        if not config_path.exists():
            raise FileNotFoundError(f"Stacking config not found: {config_path}")

        with open(config_path, 'rb') as f:
            config = pickle.load(f)

        # Stacking 인스턴스 생성
        stacking = cls(
            experimenter=experimenter,
            target_edges=config['target_edges'],
            output_var=config['output_var'],
            method=config['method'],
            include_target=config['include_target']
        )
        stacking.name = name

        return stacking

    def _load_node_data(self, node):
        """파일에서 node 데이터와 columns를 로드"""
        file_path = self.stacking_dir / f"{node}.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"Stacking data not found: {file_path}")
        with open(file_path, 'rb') as f:
            node_info = pickle.load(f)
        # columns를 메모리에 캐시
        return node_info['data'], node_info['columns']

    def _build_sort_order(self):
        all_valid_idx = np.concatenate([
            self.experimenter.valid_idx_list[i] for i in range(self.experimenter.get_n_splits())
        ])
        return np.argsort(all_valid_idx)

    def _build_target_value(self):
        target_list = []
        for idx in range(self.experimenter.get_n_splits()):
            iterator = self.experimenter.get_data_valid(idx, self.target_edges)
            aggregated = DataWrapper.simple(iterator)
            target_list.append(aggregated)

        wrapper_cls = type(target_list[0])
        return wrapper_cls.concat(target_list, axis=0)

    def _aggregate(self, iterator):
        if self.method == 'simple':
            return DataWrapper.simple(iterator)
        elif self.method == 'mean':
            return DataWrapper.mean(iterator)
        elif self.method == 'mode':
            return DataWrapper.mode(iterator)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def _get_valid(self, result_data):
        return result_data['output_valid'].select_columns(
            resolve_columns(result_data['output_valid'], self.output_var)
        )

    def _start(self, node):
        self.stacking[node] = list()

    def _stack(self, node, idx, stk):
        l = self.stacking[node]
        if len(l) != idx:
            raise RuntimeError("")
        self.stacking[node].append(
            stk.to_array()
        )
        if idx == 0:
            self.columns[node] = stk.get_columns()
    
    def _end(self, node):
        # 중간 결과를 하나의 array로 합침
        result = np.concatenate(self.stacking[node])

        # 데이터와 columns를 함께 pickle로 저장
        stacking_dir = self._ensure_stacking_dir()
        file_path = stacking_dir / f"{node}.pkl"
        node_info = {
            'data': result,
            'columns': self.columns[node]
        }
        with open(file_path, 'wb') as f:
            pickle.dump(node_info, f)

        # 메모리에서 제거
        del self.stacking[node]
        del self.columns[node]
    
    def _get_saved_nodes(self):
        """파일 시스템에서 저장된 노드 목록을 가져옴"""
        stacking_dir = self.stacking_dir
        if not stacking_dir.exists():
            return []
        return [f.stem for f in stacking_dir.glob("*.pkl") if not f.stem.startswith('__') ]

    def _get_nodes(self, nodes):
        if nodes is None:
            # 파일 시스템에서 저장된 노드 목록을 가져옴
            node_names = self._get_saved_nodes()
        elif isinstance(nodes, list):
            saved_nodes = set(self._get_saved_nodes())
            node_names = [n for n in nodes if n in saved_nodes]
        elif isinstance(nodes, str):
            pat = re.compile(nodes)
            all_nodes = self._get_saved_nodes()
            node_names = [k for k in all_nodes if k is not None and pat.search(k)]
        else:
            raise ValueError(f"nodes must be None, list, or str, got {type(nodes)}")
        return node_names

    def reset_nodes(self, nodes):
        for node in nodes:
            # 파일 삭제
            file_path = self.stacking_dir / f"{node}.pkl"
            if file_path.exists():
                file_path.unlink()
            # 메모리 캐시에서도 제거
            if node in self.columns:
                del self.columns[node]
    
    def get_dataset(self, nodes):
        node_names = self._get_nodes(nodes)
        target = self._build_target_value()
        wrapper_cls = type(target)

        # 파일에서 데이터 로드 (columns도 캐시됨)
        node_data = list()
        # columns 수집
        column_names = list()
        for i in node_names:
            data, columns = self._load_node_data(i)
            node_data.append(data)
            column_names.extend(columns)

        if self.include_target:
            return wrapper_cls.concat([
                wrapper_cls.from_output(
                    np.concatenate(node_data, axis=1),
                    column_names, target.get_index()
                )
            ] + [target], axis=1).iloc(self._build_sort_order()).data
        else:
            return wrapper_cls.from_output(
                np.concatenate(node_data, axis=1),
                column_names, target.get_index()
            ).iloc(self._build_sort_order()).data