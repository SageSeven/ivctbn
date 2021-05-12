from CTBNGenerator import *


def get_true_graph():
    true_graph_0 = pd.read_csv("../data/Ground_Truth_new.csv", index_col=1)
    true_graph = []
    for i in range(18):
        # noinspection PyBroadException
        try:
            true_graph.append(sorted(list(true_graph_0.loc[i].values.reshape(1, -1)[0])))
        except Exception:
            true_graph.append([])
    return true_graph


true_graph_list = get_true_graph()
N = 18
n_states = 2

generator = CTBNGenerator(N, n_states, "given-graph", [true_graph_list], "linear-random", [3, 1, 4])
graph = generator.get_graph()
print(graph)

str_cim_list = generator.get_str_cim_list()
print(str_cim_list)

ctbn = CTBN(graph.parent_list, n_states, generator.get_cim_list())
sample = ctbn.sample(2000)

print(sample)

learn = CTBNLearn(sample, n_states, N)
learn.maximize_likelihood(0.0, verbose=True)

print(learn.graph)
