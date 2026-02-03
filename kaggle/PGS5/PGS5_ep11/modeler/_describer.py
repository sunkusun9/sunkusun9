def desc_spec(exp):
    """실험 스펙을 Markdown으로 반환"""
    lines = []

    # 실험 타이틀
    if exp.title:
        lines.append(f"## {exp.title}")
        lines.append("")

    lines.append("| 항목 | 값 |")
    lines.append("|------|-----|")

    # Outer Splitter (sp)
    sp_name = type(exp.sp).__name__
    sp_params = []
    if hasattr(exp.sp, 'n_splits'):
        sp_params.append(f"n_splits={exp.sp.n_splits}")
    if hasattr(exp.sp, 'random_state') and exp.sp.random_state is not None:
        sp_params.append(f"random_state={exp.sp.random_state}")
    if hasattr(exp.sp, 'test_size') and exp.sp.test_size is not None:
        sp_params.append(f"test_size={exp.sp.test_size}")
    if hasattr(exp.sp, 'shuffle'):
        sp_params.append(f"shuffle={exp.sp.shuffle}")
    sp_info = f"{sp_name}({', '.join(sp_params)})" if sp_params else sp_name
    lines.append(f"| **Outer Splitter (sp)** | `{sp_info}` |")

    # Inner Splitter (sp_v)
    if exp.sp_v is not None:
        sp_v_name = type(exp.sp_v).__name__
        sp_v_params = []
        if hasattr(exp.sp_v, 'n_splits'):
            sp_v_params.append(f"n_splits={exp.sp_v.n_splits}")
        if hasattr(exp.sp_v, 'random_state') and exp.sp_v.random_state is not None:
            sp_v_params.append(f"random_state={exp.sp_v.random_state}")
        if hasattr(exp.sp_v, 'test_size') and exp.sp_v.test_size is not None:
            sp_v_params.append(f"test_size={exp.sp_v.test_size}")
        if hasattr(exp.sp_v, 'shuffle'):
            sp_v_params.append(f"shuffle={exp.sp_v.shuffle}")
        sp_v_info = f"{sp_v_name}({', '.join(sp_v_params)})" if sp_v_params else sp_v_name
        lines.append(f"| **Inner Splitter (sp_v)** | `{sp_v_info}` |")
    else:
        lines.append(f"| **Inner Splitter (sp_v)** | None |")

    # Splitter Params
    if exp.splitter_params:
        params_str = ", ".join([f"{k}='{v}'" for k, v in exp.splitter_params.items()])
        lines.append(f"| **Splitter Params** | `{{{params_str}}}` |")
    else:
        lines.append(f"| **Splitter Params** | `{{}}` |")

    # Fold 수
    lines.append(f"| **Outer Folds** | {len(exp.train_idx_list)} |")
    if len(exp.train_idx_list) > 0:
        inner_folds = len(exp.train_idx_list[0])
        lines.append(f"| **Inner Folds** | {inner_folds} |")

    return "\n".join(lines)

def desc_pipeline(exp, max_depth=None, direction='TD'):
    """파이프라인 구조를 Mermaid Markdown으로 반환

    Args:
        exp: Experimenter 인스턴스
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
        for name, node in exp.nodes.items():
            if name is not None:
                for key, edge_list in node.edges.items():
                    for edge_name, _ in edge_list:
                        if (current_node == 'Root' and edge_name is None) or (edge_name == current_node):
                            # child 노드 발견
                            if name not in node_priorities:
                                queue.append((name, priority + 1))

    # 2. Group 단위 우선순위 생성 (포함된 노드 중 가장 낮은 우선순위 = 가장 상위)
    grp_priorities = {}
    for grp_name, grp in exp.grps.items():
        if len(grp.nodes) > 0:
            grp_priorities[grp_name] = min(node_priorities.get(node_name, float('inf')) for node_name in grp.nodes)
        else:
            grp_priorities[grp_name] = float('inf')

    # 3. 소속 그룹이 없는 노드와 최상위 그룹 수집
    grouped_nodes = set()
    for grp in exp.grps.values():
        grouped_nodes.update(grp.nodes)

    ungrouped_nodes = [name for name in exp.nodes.keys() if name is not None and name not in grouped_nodes]
    top_level_items = []

    # 최상위 그룹 (parent_grp가 None인 그룹)
    for grp_name, grp in exp.grps.items():
        if grp.parent_grp is None:
            top_level_items.append(('group', grp))

    # 소속 그룹이 없는 노드들
    for node_name in ungrouped_nodes:
        if node_name in exp.nodes:
            top_level_items.append(('node', exp.nodes[node_name]))

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
                if node_name in exp.nodes:
                    items.append(('node', exp.nodes[node_name]))

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
            grp = exp.grps[top_item_name]
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
            if node_name in exp.nodes:
                node = exp.nodes[node_name]
                for key, edge_list in node.edges.items():
                    for edge_name, edge_var in edge_list:
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

    return "\n".join(lines)

def desc_node(exp, node_name, direction='TD', show_params=False):
    """특정 노드까지의 연결 구조를 Mermaid Markdown으로 반환

    Args:
        exp: Experimenter 인스턴스
        node_name: 대상 노드 이름
        direction: 그래프 방향 ('TD': Top-Down, 'LR': Left-Right)
        show_params: True이면 노드의 파라미터 정보를 표시 (default: False)
    """
    if node_name not in exp.nodes or node_name is None:
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
            for name, node in exp.nodes.items():
                if name is not None and name not in visited:
                    found = False
                    for key, edge_list in node.edges.items():
                        if found:
                            break
                        for edge_name, _ in edge_list:
                            if (current == 'Root' and edge_name is None) or (edge_name == current):
                                new_path = path + [name]
                                new_visited = visited | {name}
                                queue.append((new_path, new_visited))
                                found = True
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

    # 노드의 grp 경로를 구하는 헬퍼
    def get_grp_path(node):
        if node.grp is None:
            return node.name
        parts = []
        grp = node.grp
        while grp is not None:
            parts.insert(0, grp.name)
            grp = grp.parent_grp
        parts.append(node.name)
        return '/'.join(parts)

    # 각 노드를 subgraph로 생성
    for name in sorted(all_nodes):
        if name in exp.nodes:
            node = exp.nodes[name]

            display_name = get_grp_path(node)
            lines.append(f"    subgraph node_{name}[\"{display_name}\"]")

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

    # 경로상의 엣지 수집 (key별로 구분)
    # edges_dict: {(source, target): set of keys}
    edges_dict = {}
    for name in all_nodes:
        if name in exp.nodes:
            node = exp.nodes[name]
            for key, edge_list in node.edges.items():
                for edge_name, _ in edge_list:
                    if edge_name is None:
                        source = "Root"
                    else:
                        source = f"node_{edge_name}"
                    target = f"node_{name}"
                    # source가 경로에 포함된 경우만
                    source_node = edge_name if edge_name else 'Root'
                    if source_node in all_nodes or source_node == 'Root':
                        edge_key = (source, target)
                        if edge_key not in edges_dict:
                            edges_dict[edge_key] = set()
                        edges_dict[edge_key].add(key)

    # 엣지 출력 (key 표시)
    for (source, target), keys in sorted(edges_dict.items()):
        keys_str = ','.join(sorted(keys))
        if keys_str != 'X':
            lines.append(f"    {source} -->|{keys_str}| {target}")
        else:
            lines.append(f"    {source} --> {target}")

    lines.append("```")
    lines.append("")
    target_display = get_grp_path(exp.nodes[node_name])
    lines.append(f"**Path from Root to '{target_display}' ({len(paths)} path(s) found)**")

    # Edge 정보 테이블 추가
    target_node = exp.nodes[node_name]
    lines.append("")
    lines.append("### Edges")
    lines.append("")
    lines.append("| Key | Node | Var |")
    lines.append("|-----|------|-----|")

    for key in sorted(target_node.edges.keys()):
        edge_list = target_node.edges[key]
        for edge_name, var_spec in edge_list:
            if edge_name is None:
                node_display = "Root"
            else:
                edge_node = exp.nodes.get(edge_name)
                if edge_node:
                    node_display = get_grp_path(edge_node)
                else:
                    node_display = edge_name
            var_display = "*" if var_spec is None else f"`{var_spec}`"
            lines.append(f"| {key} | {node_display} | {var_display} |")

    return "\n".join(lines)

def desc_node_vars(exp, node_name, idx):
    """특정 노드의 입력/출력 변수를 DataFrame으로 정리

    Args:
        exp: Experimenter 인스턴스
        node_name: 대상 노드 이름
        idx: 외부 fold 인덱스

    Returns:
        tuple: (입력변수 DataFrame, 출력변수 DataFrame)
            - 입력변수 DataFrame: MultiIndex(처리노드명, 일련번호), 컬럼='name', 값=전체변수명
            - 출력변수 DataFrame: Index=일련번호, 컬럼='name', 값=전체변수명
    """
    import pandas as pd

    # get_node_vars 호출
    result = exp.get_node_vars(node_name, idx)

    if not result:
        return pd.DataFrame(columns=['name']), pd.DataFrame(columns=['name'])

    # 첫 번째 항목 사용 (가장 빈도 높은 것)
    input_vars, output_vars, fold_indices = result[0]

    # 입력 변수 DataFrame 생성
    input_data = []
    for var in input_vars:
        if '__' in var:
            node = var.split('__')[0]
        else:
            node = 'Root'
        input_data.append({'node': node, 'name': var})

    if input_data:
        input_df = pd.DataFrame(input_data)
        # 노드별로 일련번호 부여
        input_df['seq'] = input_df.groupby('node').cumcount()
        input_df = input_df.set_index(['node', 'seq'])[['name']]
    else:
        input_df = pd.DataFrame(columns=['name'])

    # 출력 변수 DataFrame 생성
    if output_vars:
        output_df = pd.DataFrame({'name': output_vars})
    else:
        output_df = pd.DataFrame(columns=['name'])

    return input_df, output_df
