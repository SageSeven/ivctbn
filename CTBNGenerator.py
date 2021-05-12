from CTBNLearn import *


class CTBNGenerator:
    def __init__(self, N, n_states, graph_method, graph_args, cim_method, cim_args):
        self.N = N
        self.n_states = n_states
        self.graph_method = graph_method
        self.graph_args = graph_args
        self.cim_method = cim_method
        self.cim_args = cim_args
        self.graph = None
        self.cim_list = None

    def get_graph(self) -> Graph:
        if self.graph is None:
            self.generate_graph()
        return self.graph

    def get_cim_list(self):
        if self.cim_list is None:
            self.generate_cim()
        return self.cim_list

    def generate_graph(self):
        if self.graph_method == "k-parents":
            k = self.graph_args[0]
            if k >= self.N:
                print("Generate graph error: k>=N.")
                k = self.N - 1
            parent_list = []
            for node in range(self.N):
                all_par = list(np.arange(self.N))
                all_par.remove(node)
                np.random.shuffle(all_par)
                parent_list.append(all_par[:k])
            self.graph = Graph(self.n_states, parent_list)
        elif self.graph_method == "given-graph":
            parent_list = self.graph_args[0]
            self.graph = Graph(self.n_states, parent_list)
        else:
            print("Generate graph error: unresolved method.")

    def generate_cim(self):
        if self.cim_method == "linear-random":
            # low: base q value; diff: random value add on low; add: extra q for each parent
            low, diff, add = self.cim_args
            graph = self.get_graph()
            self.cim_list = []
            for node in range(self.N):
                cim_list_node = []
                parents = graph.get_parents(node)
                for par_value in all_values(self.n_states, len(parents)):
                    base_q = low + sum(par_value) * add
                    cim = []
                    for x in range(self.n_states):
                        extra_q = np.random.rand() * diff
                        q = base_q + extra_q
                        line = []
                        for y in range(self.n_states):
                            if y == x:
                                line.append(0)
                            else:
                                line.append(np.random.rand())
                        line = list(np.array(line) / sum(line) * q)
                        line[x] = -q
                        cim.append(line)
                    cim_list_node.append(np.array(cim))
                self.cim_list.append(cim_list_node)
        else:
            print("Generate CIM error: unresolved method.")

    def get_str_cim_list(self):
        cim_list = self.get_cim_list()
        result = "CIM list:\n"
        for node, cim_list_node in enumerate(cim_list):
            result += "Node #" + str(node) + ":\n"
            for par_code, cim_list_node_par in enumerate(cim_list_node):
                result += "\tpar_code #" + str(par_code) + ": " + str(cim_list_node_par) + "\n"
        return result
