import numpy as np
from numpy.linalg import matrix_power
import sys
import pydot
from IPython.core.display import SVG
import math
from typing import Union

sys.setrecursionlimit(10000)

class Graph(object):
    def __init__(self, nodes: list["Node"], gtype="digraph", matrix: np.array = np.array([])):
        self.nodes = nodes
        self.gtype = gtype
        self.matrix = matrix if len(matrix) > 0 else self.matrix_from_graph()

    def show(self) -> "SVG":
        graph = pydot.Dot("graphic", graph_type=self.gtype, bgcolor="white")

        for child in self.nodes:
            graph = child.get_graph(graph)

        return SVG(data=graph.create_svg())

    def get_by_tag(self, tag) -> "Node":
        res = list(filter(lambda x: x.tag == tag, self.nodes))

        if len(res) > 0:
            return res[0]

        return None

    def matrix_from_graph(self) -> np.array:
        m = np.array([np.array([0 for _ in range(0, len(self.nodes))]) for _ in range(0, len(self.nodes))])

        for i, n1 in enumerate(self.nodes):
            for n2 in n1.nodes:
                n2_index = self.nodes.index(n2)
                m[n2_index][i] = 1

        return m

    def transitive_closure_matrix(self):
        pwed_mx = matrix_power(self.matrix, 2)

        return pwed_mx

    def to_tags(self):
        return self._to_tags(self.nodes)

    @classmethod
    def _to_tags(self, nodes: list["Node"]) -> list:
        return list(map(lambda x: x.tag, nodes))

    def dijkstra(self: 'Graph', src: 'Node', dest: 'Node') -> Union[list['Node'], int]:
        return src.dijkstra(dest, self.nodes)

class Node:
    def __init__(self, tag: str):
        self.tag: str = tag
        self.vertex_weights = []
        self.nodes: list["Node"] = []

    def get_graph(self, graph: "pydot.Dot" = None, visited: list['Node'] = []) -> Graph:
        if graph is None:
            graph = pydot.Dot("graphic", graph_type="digraph", bgcolor="white")

        graph.add_node(pydot.Node(self.tag))

        for i, child in enumerate(self.nodes):
            visited_id = "%s-%s" % (self.tag, child.tag)

            if visited_id in visited:
                continue

            graph.add_node(pydot.Node(child.tag))
            visited.append(visited_id)

            if len(graph.get_edge(self.tag, child.tag)) > 0:
                continue

            graph.add_edge(pydot.Edge(self.tag, child.tag, color="black", label=self.vertex_weights[i]))

            child.get_graph(graph, visited)

        return graph

    @staticmethod
    def min_path_weight(nodes_weights: list['int']) -> int:
        mini = math.inf

        for weight in nodes_weights:
            if weight == math.inf:
                continue

            if mini == math.inf and type(weight) == int and weight <= 0:
                mini = weight
                continue

            if mini >= float(weight) and weight <= 0:
                continue

            mini = weight

        return mini

    def dijkstra(self: 'Node', dest: 'Node', all_nodes: list['Node']) -> Union[list['Node'], int]:
        paths_weight = list(map(lambda _: math.inf, all_nodes))

        start = all_nodes.index(self)
        queue = [all_nodes[start]]
        path = []
        paths_weight[start] = 0

        while len(queue) > 0:
            # all_nodes.
            n1 = queue.pop(0)
            path.append(n1)

            n1_current_best_distance = paths_weight[all_nodes.index(n1)]
            n1_current_best_distance = n1_current_best_distance if n1_current_best_distance != math.inf else 0
            current_node_path_weights = list(map(lambda _: math.inf, all_nodes))

            for i, n2 in enumerate(n1.nodes):
                n2_pos = all_nodes.index(n2)

                # Check if node has already been visted
                if paths_weight[n2_pos] != math.inf:
                    continue

                n1_to_n2_d = n1.vertex_weights[i]
                current_node_path_weights[n2_pos] = n1_to_n2_d

                cumulative_distance_n1_to_n2 = n1_to_n2_d + n1_current_best_distance
                paths_weight[n2_pos] = cumulative_distance_n1_to_n2

            min_path_index = current_node_path_weights.index(Node.min_path_weight(current_node_path_weights))

            if min_path_index != math.inf and paths_weight[all_nodes.index(dest)] == math.inf:
                queue.append(all_nodes[min_path_index])

        path.append(dest)

        return path, paths_weight[all_nodes.index(dest)]

    def show(self) -> "SVG":
        return SVG(data=self.get_graph().create_svg())

    def is_reachable(self, n: "Node"):
        return len(self.deep_search(n)) > 0

    def deep_search(self, dest: 'Node') -> list['Node']:
        res = self.deep_path()
        res_stopping_to_dest = []

        for n in res:
            res_stopping_to_dest.append(n)

            if n.tag == dest.tag:
                break

        return res_stopping_to_dest if dest in res_stopping_to_dest else []

    def deep_path(self: "Node", visited: list["Node"] = []) -> list["Node"]:
        if self in visited:
            return visited

        visited.append(self)

        for n in self.nodes:
            visited = n.deep_path(visited)

        return visited

    def debug_nodes(self):
        Node.debug_node_list(self.nodes)

    @staticmethod
    def debug_node_list(l: list['Node']):
        print(list(map(lambda x: x.tag if x != None else "NULL", l)))


def has_index(ls, i) ->bool :
    try:
        ls[i]
        return True
        # Do something with item
    except IndexError:
        return False

def matrix_to_graph(matrix: np.array, labels: list[str] = []) -> Graph:

    if len(matrix) <= 0 and len(matrix[0]) <= 0:
        print("Empty matrix")

    res: list[Node] = []

    def label_from_row(irow: int) -> str:
        l = labels[irow] if has_index(labels, irow) else irow
        return str(l)

    for irow, _ in enumerate(matrix):
        res.append(Node(label_from_row(irow)))

    for irow, row in enumerate(matrix):
        n = res[irow]
        for icol, col in enumerate(row):
            if col >= 1:
                for _ in range(0, col):
                    res[icol].nodes.append(n)
                    res[icol].vertex_weights.append(col)

    return Graph(nodes=res, matrix=matrix)
