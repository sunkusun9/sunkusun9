import pickle as pkl
from data_wrapper import wrap, unwrap
from sklearn.model_selection import ShuffleSplit
from node import NodeGroup, Node, RootNode

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

def create_like(exp, data, data_names=None, sp=None, sp_v=None, **args):
    """기존 Experimenter의 구조를 복제하여 새로운 Experimenter 생성

    Args:
        exp: 구조를 복제할 원본 Experimenter
        data: 새로운 데이터
        data_names: 새로운 데이터의 컬럼명 (None이면 자동)
        sp: 외부 fold splitter (None이면 원본과 동일)
        sp_v: 내부 fold splitter (None이면 원본과 동일)
        **args: split에 사용할 추가 인자

    Returns:
        Experimenter: 새로 생성된 Experimenter 인스턴스
    """
    print("🔄 Creating new Experimenter with same structure...")

    # sp와 sp_v가 None이면 원본과 동일한 설정 사용
    if sp is None:
        # 원본의 split 설정을 추정 (fold 수만 맞춤)
        n_splits = len(exp.train_idx_list)
        sp = ShuffleSplit(n_splits=n_splits, random_state=1)

    # 새 Experimenter 생성
    new_exp = Experimenter(data, data_names=data_names, sp=sp, sp_v=sp_v, **args)
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
    queue = [('Root', 1)]

    while queue:
        current_node, priority = queue.pop(0)

        if current_node in node_priorities:
            continue

        node_priorities[current_node] = priority

        # current_node를 edge로 가지는 child 노드들 찾기
        for name, node in exp.nodes.items():
            if name is not None and name not in node_priorities:
                for edge_name, _ in node.edges:
                    if (current_node == 'Root' and edge_name is None) or (edge_name == current_node):
                        queue.append((name, priority + 1))
                        break

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