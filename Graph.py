import numpy as np
import copy
import json


# representing graph in CTBN
class Graph:
    def __init__(self, n_states, parent_list):
        """
        Parameters
        ----------
        :param n_states: int
            Number of states each node can take.
        :param parent_list: list of list
            Representing the adjacency matrix of CTBN. The nth element defines the parent set of the nth node.
            Example: [[1, 2], [], [1]] means a graph with 3 nodes, where node 0 has parents 1,2; node 2 has parent 1,
                and node 1 has no parent.
            Each parent set is stored as a sorted np.array.
        """
        self.n_states = n_states
        self.parent_list = parent_list.copy()
        self.N = len(self.parent_list)
        for i, plist in enumerate(self.parent_list):
            self.parent_list[i] = sorted(plist)

    def __str__(self):
        res = "Graph info:\n"
        for node in range(self.N):
            res += "Node #" + str(node) + ": " + str(self.parent_list[node]) + "\n"
        return res

    def get_parents(self, node):
        # Get parent list of the node
        # node: int, the id of the node (start from 0)
        return self.parent_list[node]

    def encode_parent_value(self, node, node_values):
        """
        Encode a node_values array to the number of CIM of node.

        :param node: int
            The number of the target node.
        :param node_values: 1-D array, value in {0, 1, ..., self.n_states - 1}, shape: (N,)
            where N is the total number of nodes.
        :return: int
            The code of the CIM of node 'node' with values 'node_values', using to get the CIM in the CIM array.

        Encoding pattern:
            Begin with the smallest number of parents, use (self.n_states) digits to encode. For example:
            node 0 with parents [1,2,4], self.n_states = 5, node_values = [0,2,1,3,4],
            then the code will be 2*3^0+1*3^1+4*3^2=39.
        """
        parents = self.get_parents(node)
        if len(parents) == 0:  # has no parents, only 1 cim
            return 0
        parent_values = np.array(node_values)[parents]
        code = 0
        for power, value in enumerate(parent_values):
            code += value * (self.n_states ** power)
        return code

    def change_edge(self, node, par):
        """
        Add edge if par is node's parent, otherwise remove it.

        :param node:
        :param par:
        :return: A new Graph object with edge changed.
        """
        new_graph = copy.deepcopy(self)
        if par in new_graph.parent_list[node]:
            new_graph.parent_list[node].remove(par)
        else:
            new_graph.parent_list[node].append(par)
            new_graph.parent_list[node].sort()
        return new_graph

    def get_likelihood(self):
        """
        Get Dim(G), the number of independent parameters in G.
        :return: int, Dim(G).
        """
        par_per_cim = self.n_states * (self.n_states - 1)
        num_cim = sum(self.n_states ** len(par_list) for par_list in self.parent_list)
        return par_per_cim * num_cim

    def to_file(self, path):
        with open(path, "w") as file:
            file.write(str(self.n_states) + "#")
            file.write(str(self.parent_list))
            file.close()

    @staticmethod
    def load_file(path):
        with open(path, "r") as file:
            s = file.read()
            arr = s.split('#')
            n_states = int(arr[0])
            parent_list = json.loads(arr[1])
            file.close()
        return Graph(n_states, parent_list)

    @staticmethod
    def generate_empty_graph(n_states, N):
        return Graph(n_states, [[]]*N)

    @staticmethod
    def get_f_score(learned, true, beta=1):
        if sum(len(arr) for arr in learned.parent_list) == 0:
            return 0
        tp = 0
        fp = 0
        fn = 0
        for node in range(learned.N):
            tp += len(set(learned.parent_list[node]).intersection(set(true.parent_list[node])))
            fp += len(set(learned.parent_list[node]) - set(true.parent_list[node]))
            fn += len(set(true.parent_list[node]) - set(learned.parent_list[node]))
        if tp == 0:
            return 0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_score = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * (precision + recall))
        return f_score
