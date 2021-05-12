from IvCTBN import *
from CTBNGenerator import *
from multiprocessing import Pool
from IvCTBNLearn import *

N = 4
n_states = 2
parent_list = [[], [0], [1], [2]]
true_graph = Graph(n_states, parent_list)
threshold = 0.0


def generate_one_sample(random_seed):
    tau = 800
    mu = 0.23
    need_len = 10000
    need_rate = 0.1

    np.random.seed(random_seed)
    while True:
        generator = CTBNGenerator(N, n_states, "given-graph", [parent_list], "linear-random", [3, 1, 4])
        iv_ctbn = IvCTBN(generator.get_graph(), generator.get_cim_list(), mu)
        sample = iv_ctbn.sample(tau)
        sample = sample[:need_len]
        sample_len = len(sample)
        iv_number = sample[N].sum()
        iv_rate = iv_number / sample_len
        print(sample_len, iv_number)

        if sample_len < need_len:
            continue
        if iv_rate < need_rate * 0.9 or iv_rate > need_rate * 1.1:
            continue
        return sample


def learn_iv(sample):
    iv_learn = IvCTBNLearn(sample, n_states, N, "empty", None, None, None)
    iv_learn.maximize_likelihood(threshold)
    learn_graph = iv_learn.graph
    f_score = Graph.get_f_score(learn_graph, true_graph)
    return f_score


def learn_trad(sample):
    learn = CTBNLearn(sample, n_states, N, "empty", None, None, None)
    learn.maximize_likelihood(threshold)
    learn_graph = learn.graph
    f_score = Graph.get_f_score(learn_graph, true_graph)
    return f_score


if __name__ == '__main__':
    pool = Pool(12)
    seeds = np.arange(100)
    samples = pool.map(generate_one_sample, seeds)
    pool.close()
    pool.join()

    iv_pool = Pool(12)
    iv_result = iv_pool.map(learn_iv, samples)
    iv_pool.close()
    iv_pool.join()

    trad_pool = Pool(12)
    trad_result = trad_pool.map(learn_trad, samples)
    trad_pool.close()
    trad_pool.join()

    result = pd.DataFrame()
    result["iv"] = iv_result
    result["trad"] = trad_result
    result.to_csv("result.csv")
