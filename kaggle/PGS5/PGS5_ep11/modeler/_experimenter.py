import uuid
import pickle as pkl
from sklearn.model_selection import ShuffleSplit

from ._data_wrapper import wrap, unwrap
from ._node import NodeGroup, Node, RootNode
from ._describer import desc_spec, desc_pipeline, desc_node, desc_node_vars

class Experimenter():
    def __init__(self, data, data_names=None, sp=ShuffleSplit(n_splits=1, random_state=1), sp_v=None, splitter_params=None, title=None):
        self.train_idx_list = list()
        self.valid_idx_list = list()
        data_native = data
        data = wrap(data)
        self.root = data

        # 실험 타이틀 저장
        self.title = title

        # splitter 설정 저장
        self.sp = sp
        self.sp_v = sp_v
        self.splitter_params = splitter_params if splitter_params is not None else {}
        self.exp_id = str(uuid.uuid4())

        split_params = {}

        if data_names is None:
            data_names = data.get_columns()
        for k, v in self.splitter_params.items():
            split_params[k] = unwrap(data.select_columns(v))

        for train_idx, valid_idx in sp.split(data_native, **split_params):
            if sp_v is not None:
                train_data = data.iloc(train_idx)
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
        self.nodes = {None: RootNode(self, data)}
        self.grps = {}

    def get_n_splits(self):
        return len(self.train_idx_list)
    
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

    def get_parents(self, node_name):
        if node_name not in self.nodes:
            return []

        node = self.nodes[node_name]
        if node.grp_name is None:
            return []

        result = []
        current_grp = self.grps.get(node.grp_name)

        while current_grp is not None:
            result.append(current_grp.name)
            current_grp = current_grp.parent_grp

        return result

    def remove_node(self, name):
        """노드를 제거

        Args:
            name: 제거할 노드 이름

        Raises:
            ValueError: 노드가 존재하지 않거나, 자식 노드가 있는 경우
        """
        # 노드가 존재하는지 확인
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' not found")

        # Root 노드는 제거 불가
        if name is None:
            raise ValueError("Cannot remove Root node")

        # 자식 노드(descendants)가 있는지 확인
        descendants = self._find_descendants(name)
        if descendants:
            descendants_list = sorted(descendants)
            raise ValueError(f"Cannot remove node '{name}': has {len(descendants)} dependent node(s): {descendants_list}")

        # 그룹에 속해있으면 그룹의 nodes 리스트에서 제거
        node = self.nodes[name]
        if node.grp_name is not None and node.grp_name in self.grps:
            grp = self.grps[node.grp_name]
            if name in grp.nodes:
                grp.nodes.remove(name)
                print(f"  ├─ Removed '{name}' from group '{node.grp_name}'")

        # nodes 딕셔너리에서 제거
        del self.nodes[name]

        print(f"✅ Node '{name}' removed")

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

                train_concat = type(train_sub[0]).concat(train_sub, axis=1)
                outer_concat = type(outer_valid_sub[0]).concat(outer_valid_sub, axis=1)
                if len(valid_sub) > 0:
                    valid_concat = type(valid_sub[0]).concat(valid_sub, axis=1)
                    yield (train_concat, valid_concat), outer_concat
                else:
                    yield (train_concat, None), outer_concat

        data_list = list()
        for node_name, var in edges:
            data_list.append(self.nodes[node_name].get_data(idx, var))
        return ret_data_func(data_list)

    def get_data_train(self, idx, edges):
        def ret_data_func(data_list):
            for z in zip(*data_list):
                train_sub, valid_sub = list(), list()
                for train_data, train_v_data in z:
                    train_sub.append(train_data)
                    if train_v_data is not None:
                        valid_sub.append(train_v_data)
                train_concat = type(train_sub[0]).concat(train_sub, axis=1)
                if len(valid_sub) > 0:
                    valid_concat = type(valid_sub[0]).concat(valid_sub, axis=1)
                    yield train_concat, valid_concat
                else:
                    yield train_concat, None

        data_list = list()
        for node_name, var in edges:
            data_list.append(self.nodes[node_name].get_data_train(idx, var))
        return ret_data_func(data_list)
    
    def get_data_valid(self, idx, edges):
        """외부 검증 데이터에 대한 처리 결과를 가져옴

        Args:
            idx: outer fold 인덱스
            edges: [(node_name, var), ...] 형태의 edge 리스트

        Yields:
            valid_concat: 각 inner fold 모델로 처리된 외부 검증 데이터 결과 (concat)
        """
        def ret_data_func(data_list):
            for z in zip(*data_list):
                outer_valid_sub = list()
                for outer_valid_data in z:
                    outer_valid_sub.append(outer_valid_data)

                # DataWrapper의 concat 사용
                outer_concat = type(outer_valid_sub[0]).concat(outer_valid_sub, axis=1)
                yield outer_concat

        data_list = list()
        for node_name, var in edges:
            data_list.append(self.nodes[node_name].get_data_valid(idx, var))
        return ret_data_func(data_list)

    def split(self, edges):
        for idx in range(len(self.train_idx_list)):
            yield self.get_data(idx, edges)
    
    def get_node_output(self, idx, node, var = None):
        if node not in self.nodes:
            raise ValueError(f"Node '{node}' not found")
        return self.nodes[node].get_data(idx, var)

    def get_node_train_output(self, idx, node, var=None):
        if node not in self.nodes:
            raise ValueError(f"Node '{node}' not found")
        return self.nodes[node].get_data_train(idx, var)

    def get_node_valid_output(self, idx, node, var=None):
        if node not in self.nodes:
            raise ValueError(f"Node '{node}' not found")
        return self.nodes[node].get_data_valid(idx, var)

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

    def desc_spec(self):
        """실험 스펙을 Markdown으로 반환"""
        return desc_spec(self)

    def desc_pipeline(self, max_depth=None, direction='TD'):
        """파이프라인 구조를 Mermaid Markdown으로 반환

        Args:
            max_depth: 최대 표시 깊이 (None이면 무제한)
            direction: 그래프 방향 ('TD': Top-Down, 'LR': Left-Right)
        """
        return desc_pipeline(self, max_depth, direction)

    def desc_node(self, node_name, direction='TD', show_params=False):
        """특정 노드까지의 연결 구조를 Mermaid Markdown으로 반환

        Args:
            node_name: 대상 노드 이름
            direction: 그래프 방향 ('TD': Top-Down, 'LR': Left-Right)
            show_params: True이면 노드의 파라미터 정보를 표시 (default: False)
        """
        return desc_node(self, node_name, direction, show_params)

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

    def save(self, filepath):
        """Experimenter 객체를 파일로 저장

        Args:
            filepath: 저장할 파일 경로
        """
        # 모든 노드의 캐시를 언로드
        print("🗑️  Unloading all node caches before saving...")
        cache_count = 0
        for name, node in self.nodes.items():
            if name is not None and hasattr(node, '_unload_cache'):
                node._unload_cache()
                cache_count += 1
        print(f"   Unloaded cache from {cache_count} node(s)")

        # Experimenter 객체를 pickle로 저장
        print(f"💾 Saving Experimenter to {filepath}...")
        with open(filepath, 'wb') as f:
            pkl.dump(self, f)

        print(f"✅ Experimenter saved successfully")

    @staticmethod
    def load(filepath):
        """파일에서 Experimenter 객체를 불러옴

        Args:
            filepath: 불러올 파일 경로

        Returns:
            Experimenter: 불러온 Experimenter 객체
        """
        print(f"📂 Loading Experimenter from {filepath}...")
        with open(filepath, 'rb') as f:
            exp = pkl.load(f)

        print(f"✅ Experimenter loaded successfully")
        print(f"   - {len(exp.nodes) - 1} node(s)")
        print(f"   - {len(exp.grps)} group(s)")
        print(f"   - {len(exp.train_idx_list)} fold(s)")

        return exp

def create_like(exp, data, data_names=None, sp=None, sp_v=None, splitter_params=None, title=None):
    """기존 Experimenter의 구조를 복제하여 새로운 Experimenter 생성

    Args:
        exp: 구조를 복제할 원본 Experimenter
        data: 새로운 데이터
        data_names: 새로운 데이터의 컬럼명 (None이면 자동)
        sp: 외부 splitter (None이면 원본과 동일)
        sp_v: 내부 splitter (None이면 원본과 동일)
        splitter_params: splitter에 전달할 파라미터 (None이면 원본과 동일)
        title: 실험 타이틀 (None이면 원본과 동일)

    Returns:
        Experimenter: 새로 생성된 Experimenter 인스턴스
    """
    print("🔄 Creating new Experimenter with same structure...")

    if sp is None:
        sp = exp.sp
    """
    if sp_v is None:
        sp_v = exp.sp_v
    if splitter_params is None:
        splitter_params = exp.splitter_params.copy()
    """
    if title is None:
        title = exp.title

    # 새 Experimenter 생성
    new_exp = Experimenter(data, data_names=data_names, sp=sp, sp_v=sp_v, splitter_params=splitter_params, title=title)
    print(f"   ├─ Created base Experimenter with {len(new_exp.train_idx_list)} fold(s)")

    # 그룹 복제 (부모-자식 관계를 유지하기 위해 위상 정렬)
    # 1. 최상위 그룹부터 BFS로 복제
    grp_mapping = {}  # 원본 그룹명 -> 새 그룹 객체

    # 최상위 그룹 찾기 (parent_grp가 None인 그룹)
    top_level_grps = [grp for grp in exp.grps.values() if grp.parent_grp is None]

    def clone_group_recursive(orig_grp, parent_grp_name=None):
        """그룹을 재귀적으로 복제"""
        new_grp = new_exp.add_grp(
            name=orig_grp.name,
            processor=orig_grp.processor,
            edges=orig_grp.edges[:],  # 리스트 복사
            X=orig_grp.X,
            y=orig_grp.y,
            method=orig_grp.method,
            parent_grp=parent_grp_name,
            adapter=orig_grp.adapter,
            params=orig_grp.params.copy()
        )
        grp_mapping[orig_grp.name] = new_grp

        # 자식 그룹들도 복제
        for child_grp in orig_grp.child_grps:
            clone_group_recursive(child_grp, parent_grp_name=orig_grp.name)

    # 최상위 그룹부터 재귀적으로 복제
    for grp in top_level_grps:
        clone_group_recursive(grp)

    print(f"   ├─ Cloned {len(exp.grps)} group(s)")

    # 노드 복제 (위상 정렬: Root부터 BFS)
    # 1. 노드의 우선순위 계산 (BFS)
    node_priorities = {}
    queue = [(None, 1)]

    while queue:
        current_node, priority = queue.pop(0)

        if current_node in node_priorities:
            continue

        node_priorities[current_node] = priority

        # current_node를 edge로 가지는 child 노드들 찾기
        for name, node in exp.nodes.items():
            if name is not None and name not in node_priorities and name == current_node:
                queue.append((name, priority + 1))

    # 우선순위 순으로 노드 정렬
    sorted_nodes = sorted(
        [(name, node) for name, node in exp.nodes.items() if name is not None],
        key=lambda x: node_priorities.get(x[0], float('inf'))
    )

    # 노드 복제
    for name, orig_node in sorted_nodes:
        if orig_node.org_attr is not None:
            org = orig_node.org_attr
            new_exp.set_node(
                name,
                grp=orig_node.grp_name,
                processor=org['processor'],
                edges=org['edges'][:] if isinstance(org['edges'], list) else org['edges'],
                X=org['X'],
                y=org['y'],
                method=org['method'],
                adapter=org['adapter'],
                params=org['params'].copy()
            )

    print(f"   └─ Cloned {len(sorted_nodes)} node(s)")
    print("✅ Structure cloning complete!")

    return new_exp