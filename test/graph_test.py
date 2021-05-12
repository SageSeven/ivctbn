from Graph import *

n_states = 2
par_list = [
    [],
    [0, 2],
    [0],
    [0, 1, 2]
]
g = Graph(n_states, par_list)
print(g)
h = g.change_edge(1, 2)
print(g)
print(h)
i = h.change_edge(0, 3)
print(g)
print(h)
print(i)

print(g.get_likelihood())
print(h.get_likelihood())
print(i.get_likelihood())

g.to_file("graph.dat")
print(Graph.load_file("graph.dat"))
