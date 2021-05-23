from IvCTBN import *
from CTBNGenerator import *
from multiprocessing import Pool
from IvCTBNLearn import *

N = 4
n_states = 2
parent_list = [[], [0], [1], [2]]
true_graph = Graph(n_states, parent_list)
threshold = 0.0

tau = 700
mu = 0.0257
need_len = 20000
need_rate = 0.5  # 0.1 mu=0.23 0.2 mu=0.108 0.3 mu=0.063 0.4 mu=0.0392 0.5 mu=0.0257


def generate_one_sample_iv(random_seed):
    np.random.seed(random_seed)
    while True:
        generator = CTBNGenerator(N, n_states, "given-graph", [parent_list], "linear-random", [3, 1, 4])
        iv_ctbn = IvCTBN(generator.get_graph(), generator.get_cim_list(), mu)
        sample_iv = iv_ctbn.sample(tau)
        sample_iv = sample_iv[:need_len]
        sample_len = len(sample_iv)
        iv_number = sample_iv[N].sum()
        iv_rate = iv_number / sample_len
        print(sample_len, iv_number)

        if sample_len < need_len:
            continue
        if iv_rate < need_rate * 0.9 or iv_rate > need_rate * 1.1:
            continue

        return sample_iv


def generate_one_sample_tr(random_seed):
    np.random.seed(random_seed)
    while True:
        generator = CTBNGenerator(N, n_states, "given-graph", [parent_list], "linear-random", [3, 1, 4])

        ctbn = CTBN(generator.get_graph().parent_list, n_states, generator.get_cim_list())
        sample = ctbn.sample(tau*2)[:need_len]
        sample_len = len(sample)
        print(sample_len)
        if sample_len < need_len:
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
    samples_iv = pool.map(generate_one_sample_iv, seeds)
    samples = pool.map(generate_one_sample_tr, seeds)
    pool.close()
    pool.join()

    iv_pool = Pool(12)
    iv_result = iv_pool.map(learn_iv, samples_iv)
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
