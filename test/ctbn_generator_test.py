from CTBNGenerator import *

N = 6
n_states = 2
k = 2

generator = CTBNGenerator(N, n_states, "k-parents", [k], "linear-random", [3, 1, 4])
graph = generator.get_graph()
print(graph)

str_cim_list = generator.get_str_cim_list()
print(str_cim_list)

ctbn = CTBN(graph.parent_list, n_states, generator.get_cim_list())
sample = ctbn.sample(1000)

print(sample)

learn = CTBNLearn(sample, n_states, N)
learn.maximize_likelihood(0.0, verbose=True)

print(learn.graph)
