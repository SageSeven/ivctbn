from CTBNLearn import *


class IvCTBNLearn(CTBNLearn):
    def __init__(self, data, n_states, N, graph_init_method="empty", graph_init_args=None,
                 graph_cache_path="graph_cache.dat", data_cache_path="data_cache.csv"):
        super().__init__(data, n_states, N, graph_init_method, graph_init_args, graph_cache_path, data_cache_path)
        self.structure = IvStructure(self.graph, data)
        self.structure.get_likelihood()

    def get_next_structures(self):
        self.next_structures = []
        for x in range(self.N):
            for y in range(self.N):
                if x == y:
                    continue
                if y in self.graph.get_parents(x):
                    continue
                graph = self.graph.change_edge(x, y)
                self.next_structures.append(IvStructure(graph, self.data, parent=self.structure, change_node=x))


class IvStructure(Structure):
    def __init__(self, graph, data, parent=None, change_node=None):
        super().__init__(graph, data, parent, change_node)
        self.stats = IvStatistics(data, graph)


class IvStatistics(Statistics):
    def stat_from_count_and_time_u(self, node, count, time, verbose=False):
        iv_index = count.columns[-5]
        count = count[count[iv_index] == 0]
        return super().stat_from_count_and_time_u(node, count, time, verbose)
