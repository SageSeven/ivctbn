from IvCTBNLearn import *
from IvCTBN import *
from CTBNGenerator import *

N = 6
n_states = 2
k = 2
generator = CTBNGenerator(N, n_states, "k-parents", [k], "linear-random", [3, 1, 4])
print(generator.get_graph())
generator.get_graph().to_file("true_graph.dat")

mu = 1
tau = 1000

iv_ctbn = IvCTBN(generator.get_graph(), generator.get_cim_list(), mu)
sample = iv_ctbn.sample(tau)
sample.to_csv("../data/iv_generator_sample.csv")

threshold = 0.0
iv_ctbn_learn = IvCTBNLearn(sample, n_states, N)
iv_ctbn_learn.maximize_likelihood(threshold, verbose=True)

graph = iv_ctbn_learn.graph
print(graph)
graph.to_file("learned_graph.dat")
