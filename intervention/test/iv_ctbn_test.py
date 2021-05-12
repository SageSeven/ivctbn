from IvCTBN import *
from CTBNLearn import *

N = 3
parent_list = [
    [],
    [0, 2],
    [0]
]
n_states = 2
cim_list = [
    [np.array([[-1, 1], [2, -2]])],
    [np.array([[-2, 2], [3, -3]]), np.array([[-4, 4], [5, -5]]),
     np.array([[-6, 6], [7, -7]]), np.array([[-8, 8], [9, -9]])],
    [np.array([[-3.5, 3.5], [2, -2]]), np.array([[-1.2, 1.2], [4, -4]])]
]
tau = 1000
mu = 1

iv_ctbn = IvCTBN(Graph(n_states, parent_list), cim_list, mu)
sample = iv_ctbn.sample(tau, verbose=False)
print(sample)
sample.to_csv("../data/iv_ctbn_sample.csv")
iv_ctbn.iv_sample.to_csv("../data/iv_sample.csv")

ctbn_learn = CTBNLearn(sample, n_states, N)
ctbn_learn.maximize_likelihood(0.0, verbose=True)

graph = ctbn_learn.graph
print(graph)
