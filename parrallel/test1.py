from datetime import datetime
from multiprocessing import Pool
import numpy as np


def time_costing_func(array, m=1):
    res = np.zeros(len(array))
    for i in range(100000):
        res += (array + m)
    return res


if __name__ == '__main__':
    pool = Pool(14)

    args = [np.arange(i, i+10) for i in range(100)]
    start_time = datetime.now()
    result = pool.map(time_costing_func, args)
    pool.close()
    pool.join()
    end_time = datetime.now()
    print("time cost: %d" % ((end_time - start_time).microseconds + (end_time - start_time).seconds * 1000000))
    # print(result)

    start_time = datetime.now()
    result_2 = [time_costing_func(arr) for arr in args]
    end_time = datetime.now()
    print("time cost: %d" % ((end_time - start_time).microseconds + (end_time - start_time).seconds * 1000000))
    # print(result_2)
