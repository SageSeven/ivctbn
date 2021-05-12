from CTBN import *
from util import *


class CTBNLearn:
    def __init__(self, data, n_states, N, graph_init_method="empty", graph_init_args=None,
                 graph_cache_path="graph_cache.dat", data_cache_path="data_cache.csv"):
        """

        :param data: 2-D DataFrame. Columns 0,1,...,N. 0 - time, 1,...,N - N nodes.
            Each row represents a change of the system with the exact time point.
        :param n_states: int.
            Number of states each node can take.
        """
        self.data = data
        self.N = N  # 4 columns: time, trans_node, begin_state, end_state (in traditional CTBN data)
        self.n_states = n_states
        self.graph = self.initialize_graph(graph_init_method, graph_init_args)
        self.graph_cache_path = graph_cache_path
        self.structure = Structure(self.graph, data)
        self.next_structures = []
        # initialize likelihood
        self.structure.get_likelihood()
        # do data cache
        if data_cache_path is not None:
            data.to_csv(data_cache_path)

    def initialize_graph(self, graph_init_method, graph_init_args):
        if graph_init_method == "empty":
            return Graph.generate_empty_graph(self.n_states, self.N)
        elif graph_init_method == "load":
            file_path = graph_init_args[0]
            return Graph.load_file(file_path)

    def maximize_likelihood(self, threshold, verbose=False):
        """
        The main learning function. Use it to get the best performed structure.

        :param threshold: float
            The minimum likelihood difference to do the iteration.
        :param verbose: bool
            Whether to print likelihood values in iteration.
        :return: No return value. The function will update self.structure and self.graph.
        """
        likelihood_updated = True
        while likelihood_updated:
            self.get_next_structures()
            likelihood_updated = self.update_structure(threshold, verbose)

    def get_next_structures(self):
        self.next_structures = []
        for x in range(self.N):
            for y in range(self.N):
                if x == y:
                    continue
                if y in self.graph.get_parents(x):
                    continue
                graph = self.graph.change_edge(x, y)
                self.next_structures.append(Structure(graph, self.data, parent=self.structure, change_node=x))

    def update_structure(self, threshold, verbose=False):
        max_likelihood = -1e10
        max_index = -1
        for index, structure in enumerate(self.next_structures):
            if self.structure.change_node is not None and structure.change_node < self.structure.change_node:
                continue
            if structure.get_likelihood() > max_likelihood:
                max_likelihood = structure.get_likelihood()
                max_index = index
                if max_likelihood > self.structure.get_likelihood() + threshold:
                    break  # greedy setting, find the first structure which has a larger likelihood
        if verbose:
            print("prev likelihood: %.6lf, next likelihood: %.6lf"
                  % (self.structure.get_likelihood(), max_likelihood))
        if max_likelihood > self.structure.get_likelihood() + threshold:
            self.structure = self.next_structures[max_index]
            self.graph = self.structure.graph
            if self.graph_cache_path is not None:
                self.graph.to_file(self.graph_cache_path)
            return True
        else:
            return False


class Structure:
    def __init__(self, graph, data, parent=None, change_node=None):
        self.graph = graph
        self.stats = Statistics(data, graph)
        self.data = data
        self.n_states = graph.n_states
        self.likelihood_list = [None] * self.graph.N
        self.likelihood = None
        self.parent = parent
        self.change_node = change_node

    def get_likelihood(self, verbose=False):
        if self.likelihood is None:
            self.calculate_likelihood(verbose)
        return self.likelihood

    def calculate_likelihood(self, verbose=False):
        # data size: number of rows
        likelihood = -np.log(len(self.data)) * self.graph.get_likelihood() / 2
        # then calculate log-likelihood under sufficient statistics
        likelihood += self.get_param_likelihood(verbose)
        self.likelihood = likelihood

    def get_param_likelihood(self, verbose=False):
        likelihood = 0.0
        for node in range(self.graph.N):
            if self.likelihood_list[node] is not None:
                likelihood += self.likelihood_list[node]
            elif self.parent is None or node == self.change_node:
                self.likelihood_list[node] = 0.0
                node_stat = self.stats.get_statistics_node(node)
                for par_node_stat in node_stat:
                    for x, (m_trans, m_total, t) in enumerate(par_node_stat):
                        if m_total == 0:
                            continue  # no transition from x, likelihood + 0
                        q = m_total / t
                        theta = np.array(m_trans) / m_total
                        self.likelihood_list[node] += (m_total * np.log(q) - q * t)
                        if verbose:
                            print("q: %lf, theta: %s" % (q, theta))
                        for y in range(self.n_states):
                            if x == y:
                                continue
                            self.likelihood_list[node] += m_trans[y] * np.log(theta[y])
                likelihood += self.likelihood_list[node]
            else:
                self.likelihood_list[node] = self.parent.likelihood_list[node]
                likelihood += self.likelihood_list[node]

        return likelihood


class Statistics:
    def __init__(self, data, graph):
        """

        :param data: 2-D DataFrame

        Statistics structure:
        [
            [node1: [par1: ...], [par2: ...], ..., [par_n: ...]],
            [node2: ...],
            ...,
            [node_n: ...]
        ]
        For each node with each parent value:
        [par1:
            [x=0:
                [M[0->0|u], M[0->1|u], ..., M[0->n_states-1|u]],
                M[0|u],
                T[0|u]
            ],
            [x=1: ...],
            ...,
            [x=n_states-1: ...]
        ]
        where par1, par2, ..., par_n are encoded by util.all_values.
        """
        self.data = data
        self.graph = graph
        self.N = graph.N
        self.n_states = graph.n_states
        self.statistics = None

    def get_statistics(self, verbose=False):
        if self.statistics is None:
            self.calculate_statistics(verbose)
        return self.statistics

    def get_statistics_node(self, node, verbose=False):
        node = int(node)
        if self.statistics is None:
            self.statistics = [None] * self.N
        if self.statistics[node] is None:
            self.statistics[node] = self.calculate_statistics_node(node, verbose)
        return self.statistics[node]

    def calculate_statistics(self, verbose=False):
        # calculate statistics and store at self.statistics
        self.statistics = []
        for node in range(self.N):
            node_stat = self.calculate_statistics_node(node, verbose)
            self.statistics.append(node_stat)

    def calculate_statistics_node(self, node, verbose=False):
        """
        Calculate statistics where x is the value of node 'node'.

        :param node:
        :param verbose:
        :return:
        """
        node_index = self.data.columns[-3]  # last 3 indexes: node to transition, current state, next state
        count_data = self.data[self.data[node_index] == node]
        parents = self.graph.get_parents(node)  # list of parents
        node_stat = []
        # has no parent
        if len(parents) == 0:
            node_stat.append(self.stat_from_count_and_time_u(node, count_data, self.data, verbose))
        # has parents
        else:
            for cim_index, par_value in enumerate(all_values(self.n_states, len(parents))):
                count_on_u = multi_col_equal(count_data, parents, par_value)
                time_on_u = multi_col_equal(self.data, parents, par_value)
                if verbose:
                    print("node: %d, parents: %s" % (node, par_value))
                node_stat.append(self.stat_from_count_and_time_u(node, count_on_u, time_on_u, verbose))
        return node_stat

    def stat_from_count_and_time_u(self, node, count, time, verbose=False):
        # return stats based on fixed u
        result = []
        begin_index = count.columns[-2]
        emd_index = count.columns[-1]
        time_index = count.columns[-4]  # time data stored at the last 4 column
        for x in range(self.n_states):
            # structure of result_x:
            #   1. a list of M[xx'|u], where x' from 0 to n_states-1.
            #   2. M[x|u].
            #   3. T[x|u].
            result_x = []
            result_x_m = []
            count_x = count[count[begin_index] == x]
            for y in range(self.n_states):
                if y == x:
                    result_x_m.append(0)
                    continue
                count_y = count_x[count_x[emd_index] == y]
                result_x_m.append(len(count_y))
                if verbose:
                    print("node: %d, x: %d, y: %d, count: %d" % (node, x, y, len(count_y)))
            # M[xx'|u]
            result_x.append(result_x_m)
            # M[x|u]
            result_x.append(sum(result_x_m))
            # Time data: raw data conditioned with parents = u
            time_x = time[time[node] == x]
            if time_x[time_index].sum() < 0:
                print(time)
            # T[x|u]
            result_x.append(time_x[time_index].sum())
            result.append(result_x)
        return result

    def __str__(self):
        result = "Statistics info:\n"
        result += str(self.graph)
        result += "Statistics:\n"
        self.statistics = None
        for node, node_stat in enumerate(self.get_statistics()):
            result += "Node #" + str(node) + ":\n"
            for par_code, par_node_stat in enumerate(node_stat):
                result += "\tpar_code #" + str(par_code) + ":\n"
                for x, (m_trans, m_total, t) in enumerate(par_node_stat):
                    result += "\t\tstate: " + str(x) + ": "
                    result += "M[xx'|u]: " + str(m_trans) + "; "
                    result += "M[x|u]:" + str(m_total) + "; "
                    result += "T[x|u]:" + str(t) + "."
                    result += "\n"
        return result
