a = [[2, 1], [3, 2], [1, 2, 3]]
for i, b in enumerate(a):
    a[i] = sorted(b)
print(a)

import numpy as np

b = np.array([5, 6, 7, 8, 9, 10])
print(b[a[2]])

print([0] * 5)

print(np.hstack((10, [0, 1, 0])))

print(np.random.choice(6, 1, p=[0.9, 0.02, 0.01, 0.03, 0.02, 0.02]))

import scipy.stats as st

print(["%.2f" % st.expon.rvs(size=1, scale=10)[0] for i in range(20)])

a = [2, 3, 1, 4, 5]
print("2 in a:", 2 in a)
print(a.remove(2))
print("2 in a:", 2 in a)
print(a)
a.sort()
print(a)


class A:
    def __init__(self, a1):
        self.a1 = a1

    def __str__(self):
        return str(self.a1)

    def newA(self, a2):
        return A(self.a1 + a2)


a = A(2)
print("a:", a)
b = a.newA(3)
print("b:", b)

a = np.array([1, 2, 3])
a[1] = 4
print(a)
a = np.delete(a, 2)
print(a)

print(2 == 2.0)

for i, a in enumerate([]):
    print(i, a)

import pandas as pd

df = pd.DataFrame([[1, 2, 3, 4], [5, 2, 7, 8], [9, 10, 3, 12], [6, 2, 3, 8],
                   [3, 4, 5, 6], [2, 2, 2, 2], [9, 2, 3, 19], [3, 3, 3, 3]])
print(df)
df1 = df[[1, 2, 3]] == [2, 3, 4]
df2 = df1.astype(int).T.cumprod().T.astype(bool)
print(df[df2[df2.columns[-1]]])


def test1(x):
    aa = 5
    while aa < x:
        yield aa
        aa += 1


for i, a in enumerate(test1(10)):
    print(i, a)

print([0]*0)

from util import *

for i, a in enumerate(all_values(3, 4)):
    print(i, a)

df = pd.DataFrame([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
cols = [1, 2]
values = [2, 3]
print(multi_col_equal(df, cols, values))
print([[]]*10)

a, b, c = [1, 2, 3]
print(a, b, c)


class AA:
    def __init__(self):
        self.a = 1

    def p(self):
        print(self.a)


class AB(AA):
    def __init__(self):
        super(AB, self).__init__()
        self.a = 3
        self.b = 2

    def p(self):
        print(self.b)
        super(AB, self).p()


AB().p()

with open("test1.dat", "w") as file:
    file.write(str([1, [3, 4]]))
    # file.write(str([2, 3, 1, 4]))
    file.close()

import json

with open("test1.dat", "r") as file:
    s = file.read()
    print(json.loads(s))

iv_sample = pd.read_csv("../intervention/data/iv_ctbn_sample.csv", index_col=0)
print(iv_sample)

container = np.zeros((3, 4))
container[1] = [1, 2, 3, 4]
print(container)
print(container[:2])

new_container = np.zeros((6, 4))
new_container[:3] = container[:3]
container = new_container
print(new_container)

a = [None] * 10
print(a[2] is None)
print(a)

from Graph import Graph
graph = Graph.generate_empty_graph(2, 8)
graph.to_file("test_graph.dat")

print({1, 2, 3, 4} - {2, 3, 4, 5})

df = pd.DataFrame()
df["a"] = [1, 2, 3, 4, 5]
print(df)

print(df[:2])

print(str(0.1))
