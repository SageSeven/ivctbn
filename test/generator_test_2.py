from CTBNGenerator import *

N = 4
n_states = 2
parent_list = [[], [0], [1], [2]]
true_graph = Graph(n_states, parent_list)
SEED = 5

np.random.seed(SEED)
generator_1 = CTBNGenerator(N, n_states, "given-graph", [parent_list], "linear-random", [3, 1, 4])

print(generator_1.get_cim_list())

np.random.seed(SEED)
generator_2 = CTBNGenerator(N, n_states, "given-graph", [parent_list], "linear-random", [3, 1, 4])

print(generator_2.get_cim_list())
