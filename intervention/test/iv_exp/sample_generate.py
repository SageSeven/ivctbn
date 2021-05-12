from IvCTBN import *
from CTBNGenerator import *

N = 8
n_states = 2
k = 4
generator = CTBNGenerator(N, n_states, "k-parents", [k], "linear-random", [3, 1, 4])
graph = generator.get_graph()
graph.to_file("true_graph_6.dat")

tau = 2000
mu = 0.002
iv_ctbn = IvCTBN(graph, generator.get_cim_list(), mu)
sample = iv_ctbn.sample(tau)
sample.to_csv("iv_sample_6.csv")

print("Sample length: %d" % (len(sample)))
print("Intervention number: %d" % (sample[N].sum()))
