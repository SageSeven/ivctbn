from CTBNLearn import *

N = 8
n_states = 2
data = pd.read_csv("iv_sample_5.csv", index_col=0)
data.columns = list(np.arange(N + 5))

threshold = 0.0
learn = CTBNLearn(data, n_states, N, data_cache_path=None)
learn.structure.change_node = 0
learn.maximize_likelihood(threshold, verbose=True)
learn_graph = learn.graph
learn_graph.to_file("learn_graph_trad_5.dat")
print(learn_graph)
