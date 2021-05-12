from IvCTBNLearn import *

N = 8
n_states = 2
data = pd.read_csv("iv_sample_5.csv", index_col=0)
data.columns = list(np.arange(N + 5))

threshold = 0.0
iv_learn = IvCTBNLearn(data, n_states, N, graph_init_method="load", graph_init_args=["graph_cache.dat"],
                       data_cache_path=None)
iv_learn.structure.change_node = 0
iv_learn.maximize_likelihood(threshold, verbose=True)
learn_graph = iv_learn.graph
learn_graph.to_file("learn_graph_5.dat")
print(learn_graph)
