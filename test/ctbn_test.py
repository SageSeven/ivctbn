from CTBNLearn import *

np.random.seed(0)

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
ctbn = CTBN(parent_list, n_states, cim_list)
print(ctbn.get_CIMs([0, 1, 1]))
print(ctbn.graph.get_likelihood())
sample = ctbn.sample(800, verbose=False)

test1 = False
if test1:

    print(sample.columns)
    print(len(sample.columns) - 1)
    print(len(sample))
    print(sample[np.arange(3)])

    print(sample[sample[4] == 0])
    print(sample[3].sum())

print(sample)
statistic = Statistics(sample, ctbn.graph)
print(statistic.get_statistics(verbose=False))

structure = Structure(ctbn.graph, sample)
likelihood = structure.get_likelihood(verbose=True)
print(likelihood)

learn = CTBNLearn(sample, n_states, N)
threshold = 0.0
learn.maximize_likelihood(threshold, verbose=True)
learned_graph = learn.graph
print(learned_graph)

print(learn.structure.stats)
