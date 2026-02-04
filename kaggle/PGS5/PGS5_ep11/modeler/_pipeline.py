import re
from ._describer import desc_pipeline, desc_node

class PipelineGroup:
    def __init__(
        self, name, role, processor=None, edges=None, X=None, y=None,
        method=None, parent=None, adapter='default', params=None
    ):
        self.name = name
        self.role = role  # 'stage' or 'head'
        self.processor = processor
        self.edges = edges if edges is not None else {}
        self.X = X
        self.y = y
        self.method = method
        self.parent = parent  # parent group name (str)
        self.adapter = adapter
        self.params = params if params is not None else {}
        self.children = []  # child group names
        self.nodes = []  # node names in this group
        self.attrs = None

    def get_attrs(self, grps):
        if self.attrs is not None:
            return self.attrs
        if self.parent is None:
            parent_attrs = {
                'edges': {},
                'params': {},
                'processor': None,
                'X': None,
                'y': None,
                'method': None,
                'adapter': 'default'
            }
        else:
            parent_attrs = grps[self.parent].get_attrs(grps)
        edges = self.edges.copy()
        if parent_attrs['edges'] is not None:
            for k, v in parent_attrs['edges'].items():
                edges[k] = edges.get(k, []) + v
        params = self.params.copy()
        if parent_attrs['params'] is not None:
            for k, v in parent_attrs['params'].items():
                if k not in params:
                    params[k] = v
        self.attrs = {
            'name': self.name,
            'edges': edges,
            'parent': self.parent,
            'adapter': self.adapter,
            'params': params,
            'children': self.children,
        }
        for i in ['role', 'processor', 'X', 'y', 'method']:
            self.attrs[i] = parent_attrs.get(i) if getattr(self, i) is None else getattr(self, i)

        return self.attrs

    def update_attrs(self):
        self.attrs = None

    def copy(self):
        ret = PipelineGroup(
            self.name, self.role, self.processor, self.edges.copy(), self.X, self.y,
            self.method, self.parent, self.adapter, self.params.copy()
        )
        ret.children = self.children.copy()
        ret.nodes = self.nodes.copy()
        return ret


class PipelineNode:
    def __init__(
        self, name, grp, processor=None, edges=None, X=None, y=None,
        method=None, adapter='default', params=None
    ):
        self.name = name
        self.grp = grp  # group name (str)
        self.processor = processor
        self.edges = edges if edges is not None else {}
        self.X = X
        self.y = y
        self.method = method
        self.adapter = adapter
        self.params = params if params is not None else {}

        self.output_edges = []  # 이 노드를 입력으로 사용하는 노드들의 이름
        self.attrs = None

    def copy(self):
        ret = PipelineNode(
            self.name, self.grp, self.processor, self.edges.copy(), self.X, self.y,
            self.method, self.adapter, self.params.copy()
        )
        ret.output_edges = self.output_edges.copy()
        return ret

    def get_attrs(self, grps):
        if self.attrs is not None:
            return self.attrs
        grp_attrs = grps[self.grp].get_attrs(grps)
        edges = self.edges.copy()
        if grp_attrs['edges'] is not None:
            for k, v in grp_attrs['edges'].items():
                edges[k] = edges.get(k, []) + v
        params = self.params.copy()
        if grp_attrs['params'] is not None:
            for k, v in grp_attrs['params'].items():
                if k not in params:
                    params[k] = v
        self.attrs = {
            'name': self.name,
            'grp': self.grp,
            'edges': edges,
            'adapter': self.adapter,
            'params': params,
        }
        for i in ['role', 'processor', 'X', 'y', 'method']:
            self.attrs[i] = grp_attrs.get(i) if getattr(self, i) is None else getattr(self, i)

        return self.attrs

    def update_attrs(self):
        self.attrs = None


class Pipeline:
    def __init__(self):
        self.nodes = {}
        self.grps = {}

    def _validate_name(self, name):
        if name is None:
            return

        if '__' in name:
            raise ValueError(f"Name '{name}' cannot contain '__'")

        invalid_chars = ['/', '\\', '\0', '<', '>', ':', '"', '|', '?', '*']
        for char in invalid_chars:
            if char in name:
                raise ValueError(f"Name '{name}' cannot contain '{char}'")

    def _find_descendants(self, node_name):
        descendants = set()
        queue = [node_name]

        while queue:
            current = queue.pop(0)

            if current not in self.nodes:
                continue

            for child_name in self.nodes[current].output_edges:
                if child_name not in descendants:
                    descendants.add(child_name)
                    queue.append(child_name)

        return descendants

    def _check_cycle(self, node_name, new_edges):
        descendants = self._find_descendants(node_name)

        cycle_edges = []
        for key, edge_list in new_edges.items():
            for edge_name, _ in edge_list:
                if edge_name is None:
                    continue

                if edge_name not in self.nodes:
                    continue

                if edge_name in descendants:
                    cycle_edges.append(edge_name)

        if cycle_edges:
            return True, cycle_edges
        return False, []

    def _check_edges(self, edges):
        if edges is None or len(edges) == 0:
            return False
        for key, edge_list in edges.items():
            for name, _ in edge_list:
                if name is None:
                    continue
                if name not in self.nodes:
                    raise ValueError(f"Edge node '{name}' not found")
                node_grp = self.nodes[name].grp
                if self.grps[node_grp].role != 'stage':
                    raise ValueError(f"Edge node '{name}' must be a stage node, got '{self.grps[node_grp].role}'")
        return True

    def _get_all_nodes_in_grp(self, grp):
        result = list(grp.nodes)
        for child_name in grp.children:
            child_grp = self.grps[child_name]
            result.extend(self._get_all_nodes_in_grp(child_grp))
        return result

    def _get_effected_nodes(self, nodes):
        priorities = {}
        queue = []

        for node_name in nodes:
            priorities[node_name] = 1
            queue.append((node_name, 1))

        while queue:
            current_node, current_priority = queue.pop(0)

            descendants = self._find_descendants(current_node)

            for desc_node in descendants:
                new_priority = current_priority + 1
                if desc_node not in priorities or priorities[desc_node] < new_priority:
                    priorities[desc_node] = new_priority
                    queue.append((desc_node, new_priority))

        sorted_nodes = sorted(priorities.items(), key=lambda x: x[1])
        return [self.nodes[i[0]] for i in sorted_nodes]

    def set_grp(
            self, name, role=None, processor=None, edges=None, X=None, y=None, method=None, parent=None, adapter=None, params=None, replace=False
        ):
        self._validate_name(name)
        if name in self.nodes:
            raise ValueError(f"Name '{name}' already exists as a node")
        if edges is None:
            edges = {}

        if parent is not None:
            if parent not in self.grps:
                raise ValueError(f"Parent group '{parent}' not found")
            if role is None:
                role = self.grps[parent].role
        if role not in ['stage', 'head']:
            raise ValueError(f"Role must be 'stage' or 'head', got '{role}'")

        if name not in self.grps:
            self._check_edges(edges)
            grp = PipelineGroup(
                name, role, processor=processor, edges=edges, X=X, y=y, method=method, parent=parent, adapter=adapter, params=params
            )

            if parent is not None:
                self.grps[parent].children.append(name)

            self.grps[name] = grp
            return {
                "result": "new", "obj": grp
            }
        elif not replace:
            raise ValueError(f"Group '{name}' already exists. Use replace=True to update.")

        old_grp = self.grps[name]
        if old_grp.role != role:
            raise ValueError(f"Cannot change role of group '{name}': existing '{old_grp.role}', requested '{role}'")
        grp = old_grp.copy()

        parent_changed = False
        old_parent = old_grp.parent

        if old_parent != parent:
            parent_changed = True
            if old_parent is not None:
                self.grps[old_parent].children.remove(name)
            grp.parent = parent
            if parent is not None:
                self.grps[parent].children.append(name)

        if processor is not None:
            grp.processor = processor
        if edges is not None and len(edges) > 0:
            grp.edges = edges
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

        grp.update_attrs()
        attrs = grp.get_attrs(self.grps)
        new_edges = attrs['edges']
        if len(new_edges) > 0:
            affected_nodes = self._get_all_nodes_in_grp(grp)

            for node_name in affected_nodes:
                if node_name not in self.nodes:
                    continue

                node = self.nodes[node_name]
                node.update_attrs()
                node_attrs = node.get_attrs(self.grps)
                node_own_edges = node_attrs.get('edges', {})

                final_edges = {k: list(v) for k, v in new_edges.items()}
                for k, v in node_own_edges.items():
                    if k in final_edges:
                        final_edges[k].extend(v)
                    else:
                        final_edges[k] = list(v)

                has_cycle, cycle_edges = self._check_cycle(node_name, final_edges)
                if has_cycle:
                    cycle_info = ", ".join([f"'{e}'" for e in cycle_edges])
                    raise ValueError(f"Cannot update group '{name}': node '{node_name}' would create cycle through edge(s) {cycle_info}")
        else:
            affected_nodes = list()

        self.grps[name] = grp
        return {
            "result": "update", "affected_nodes": affected_nodes, "old_grp": old_grp, "grp": grp
        }

    def rename_grp(self, name_from, name_to):
        self._validate_name(name_to)

        if name_from not in self.grps:
            raise ValueError(f"Group '{name_from}' not found")
        if name_to in self.grps:
            raise ValueError(f"Group '{name_to}' already exists")

        old_grp = self.grps[name_from]
        grp = old_grp.copy()
        grp.name = name_to
        if grp.parent is not None:
            self.grps[grp.parent].children.remove(name_from)
            self.grps[grp.parent].children.append(name_to)

        for node_name in grp.nodes:
            self.nodes[node_name].grp = name_to
            self.nodes[node_name].update_attrs()

        for child_name in grp.children:
            self.grps[child_name].parent = name_to
            self.grps[child_name].update_attrs()

        del self.grps[name_from]
        self.grps[name_to] = grp

    def remove_grp(self, name):
        if name not in self.grps:
            raise ValueError(f"Group '{name}' not found")

        grp = self.grps[name]

        if len(grp.children) > 0:
            raise ValueError(f"Cannot remove group '{name}': has {len(grp.children)} child group(s)")

        if len(grp.nodes) > 0:
            raise ValueError(f"Cannot remove group '{name}': has {len(grp.nodes)} node(s)")

        if grp.parent is not None:
            self.grps[grp.parent].children.remove(name)

        del self.grps[name]

    def get_parents(self, node_name):
        if node_name not in self.nodes:
            return []

        node = self.nodes[node_name]
        if node.grp is None:
            return []

        result = []
        current_grp = self.grps.get(node.grp)

        while current_grp is not None:
            result.append(current_grp.name)
            current_grp = self.grps.get(current_grp.parent) if current_grp.parent else None

        return result

    def get_node_names(self, query):
        if isinstance(query, str):
            if query not in self.grps:
                return []

            result = []
            def collect_nodes(grp):
                result.extend(grp.nodes)
                for child_name in grp.children:
                    collect_nodes(self.grps[child_name])

            collect_nodes(self.grps[query])
            return result

        elif isinstance(query, re.Pattern):
            return [name for name in self.nodes.keys() if name is not None and query.search(name)]

        else:
            raise ValueError(f"query must be str or re.Pattern, got {type(query)}")

    def remove_node(self, name):
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' not found")

        if name is None:
            raise ValueError("Cannot remove Root node")

        descendants = self._find_descendants(name)
        if descendants:
            descendants_list = sorted(descendants)
            raise ValueError(f"Cannot remove node '{name}': has {len(descendants)} dependent node(s): {descendants_list}")

        node = self.nodes[name]

        self._update_output_edges(name, node.edges, None)

        grp_name = node.grp
        if grp_name is not None and grp_name in self.grps:
            grp = self.grps[grp_name]
            if name in grp.nodes:
                grp.nodes.remove(name)

        del self.nodes[name]

    def _update_output_edges(self, node_name, old_edges, new_edges):
        if old_edges is not None:
            for key, edge_list in old_edges.items():
                for edge_name, _ in edge_list:
                    if edge_name in self.nodes:
                        parent_node = self.nodes[edge_name]
                        if node_name in parent_node.output_edges:
                            parent_node.output_edges.remove(node_name)

        if new_edges is not None:
            for key, edge_list in new_edges.items():
                for edge_name, _ in edge_list:
                    if edge_name in self.nodes:
                        parent_node = self.nodes[edge_name]
                        if node_name not in parent_node.output_edges:
                            parent_node.output_edges.append(node_name)

    def set_node(
        self, name, grp, processor=None, edges=None, X=None, y=None,
        method=None, adapter='default', params=None, replace=False
    ):
        self._validate_name(name)

        if name in self.grps:
            raise ValueError(f"Name '{name}' already exists as a group")

        if grp not in self.grps:
            raise ValueError(f"Group '{grp}' not found")

        if edges is None:
            edges = {}
        if params is None:
            params = {}

        self._check_edges(edges)

        is_update = name in self.nodes
        if not replace and is_update:
            raise ValueError(f"Node '{name}' already exists. Use replace=True to update.")

        old_edges = None
        old_output_edges = None
        old_node = None
        if is_update:
            old_node = self.nodes[name]
            old_edges = old_node.edges
            old_output_edges = old_node.output_edges

        node = PipelineNode(
            name, grp, processor, edges, X=X, y=y, method=method, adapter=adapter, params=params
        )

        grp_obj = self.grps[grp]
        attrs = node.get_attrs(self.grps)

        if attrs.get('processor') is None:
            raise ValueError(f"Cannot create node '{name}': processor is required")

        if attrs.get('method') is None:
            raise ValueError(f"Cannot create node '{name}': method is required")

        if len(attrs.get('edges', {})) == 0:
            raise ValueError(f"Cannot create node '{name}': edges is required")

        has_cycle, cycle_edges = self._check_cycle(name, attrs['edges'])
        if has_cycle:
            cycle_info = ", ".join([f"'{e}'" for e in cycle_edges])
            raise ValueError(f"Cannot add node '{name}': would create cycle through edge(s) {cycle_info}")

        self._update_output_edges(name, old_edges, edges)

        if old_output_edges is not None:
            node.output_edges = old_output_edges

        if name not in grp_obj.nodes:
            grp_obj.nodes.append(name)

        if is_update:
            affected_nodes = list(self._find_descendants(name))
            old_grp_name = old_node.grp
            if old_grp_name != grp and old_grp_name in self.grps:
                old_grp = self.grps[old_grp_name]
                if name in old_grp.nodes:
                    old_grp.nodes.remove(name)
        else:
            affected_nodes = list()

        self.nodes[name] = node

        return {
            'result': 'update' if is_update else 'new',
            'affected_nodes': affected_nodes,
            'old_obj': old_node,
            'obj': node
        }

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

    def get_node_names(self, query):
        if isinstance(query, str):
            if query not in self.grps:
                return []

            result = []
            def collect_nodes(grp):
                result.extend(grp.nodes)
                for child_grp in grp.child_grps:
                    collect_nodes(child_grp)

            collect_nodes(self.grps[query])
            return result

        elif isinstance(query, re.Pattern):
            return [name for name in self.nodes.keys() if name is not None and query.search(name)]

        else:
            raise ValueError(f"query must be str or re.Pattern, got {type(query)}")

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