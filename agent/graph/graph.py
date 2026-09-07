from typing import List
from uuid import UUID

from agent.graph.node import Node, VoidNode


class LLMAppGraph:

    def __init__(self):
        entry_node, exit_node = VoidNode(name="entry_node"), VoidNode(name="exit_node")
        self.entry_node_uuid = entry_node.uuid
        self.exit_node_uuid = exit_node.uuid
        self.nodes = {entry_node.uuid: entry_node, exit_node.uuid: exit_node}
        self.edges = {}

    def add_node(self, node: Node):
        self.nodes[node.uuid] = node

    def add_edge(self, src: UUID, dst: UUID):
        assert isinstance(src, UUID) and isinstance(dst, UUID), "Source and destination must be UUIDs."
        assert src in self.nodes and dst in self.nodes, "Source and destination nodes must exist in the graph."
        assert src != dst, "Source and destination nodes must be different currently."
        if src not in self.edges:
            self.edges[src] = []
        self.edges[src].append(dst)

    def successors(self, node_uuid: UUID) -> List[UUID]:
        return self.edges[node_uuid] if node_uuid in self.edges else []

    def predecessors(self, node_uuid: UUID) -> List[UUID]:
        return [src for src in self.edges if node_uuid in self.edges[src]]

    def get_node(self, node_uuid: UUID) -> Node:
        assert node_uuid in self.nodes, "The node must exist in the graph."
        return self.nodes[node_uuid]


class InDegreePriorityGraph(LLMAppGraph):
    def __init__(self):
        super().__init__()
        self.in_degree_dict = {}
        self.out_degree_dict = {}
        self.depth_type_dict = {}

    def analyse_depth_info(self):
        node_depth_dict:dict = {self.entry_node_uuid: 0}

        bfs_set = set([self.entry_node_uuid])
        while bfs_set:
            node_uuid = bfs_set.pop()
            son_nodes = self.successors(node_uuid)
            for son_node_uuid in son_nodes:
                if son_node_uuid not in node_depth_dict:
                    node_depth_dict[son_node_uuid] = node_depth_dict[node_uuid] + 1
                    bfs_set.add(son_node_uuid)
        for node_uuid in self.nodes:
            self.nodes[node_uuid].add_kvargs(depth=node_depth_dict[node_uuid])

    def analyse_degree_info(self):
        for node_uuid in self.nodes:
            node = self.nodes[node_uuid]
            self.in_degree_dict[node_uuid] = len(self.predecessors(node_uuid))
            self.out_degree_dict[node_uuid] = len(self.successors(node_uuid))
            node.add_kvargs(in_degree=self.in_degree_dict[node_uuid], out_degree=self.out_degree_dict[node_uuid])

    def analyse_similarity_info(self):
        max_depth = max(self.nodes[node_uuid].kvargs.get("depth", 0) for node_uuid in self.nodes)
        self.depth_type_dict = {depth: {} for depth in range(0, max_depth + 1)}

        for node_uuid in self.nodes:
            node_depth = self.nodes[node_uuid].kvargs.get("depth", 0)
            node_type = self.nodes[node_uuid].node_type

            if node_type not in self.depth_type_dict[node_depth]:
                self.depth_type_dict[node_depth][node_type] = []
            self.depth_type_dict[node_depth][node_type].append(node_uuid)

        for depth in self.depth_type_dict:
            for node_type in self.depth_type_dict[depth]:
                node_list = self.depth_type_dict[depth][node_type]
                for node_uuid in node_list:
                    node = self.nodes[node_uuid]
                    similarity = 1 / len(node_list)
                    node.add_kvargs(similarity=similarity)

    def simple_preiority_score(self, node: Node):
        return node.kvargs.get("depth", 0)

    def set_priorities(self):
        self.analyse_depth_info()
        self.analyse_degree_info()
        self.analyse_similarity_info()

        for node_uuid in self.nodes:
            node = self.nodes[node_uuid]
            node_priority = self.simple_preiority_score(node)
            node.add_kvargs(priority=node_priority)
