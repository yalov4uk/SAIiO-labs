import numpy as np


def main():
    '''
    task 5
    '''
    g = load_graph('task5.txt')
    print min_path_tree(g, 0)


def load_graph(filename):
    with open(filename, 'r') as f:
        vertices_count = int(f.readline())
        graph = [[] for _ in xrange(vertices_count)]
        for line in f:
            from_, to_, weight = map(int, line.split())
            graph[from_].append((to_, weight))
    return graph


def min_path_tree(G, start_vertex):
    n = len(G)
    V = set(range(n))
    I_marked = set()
    B = [np.inf for _ in xrange(n)]
    f = [None for _ in xrange(n)]
    B[start_vertex] = 0
    while I_marked != V:
        v1 = min(V - I_marked, key=lambda i: B[i])
        for v2, weight in G[v1]:
            if B[v1] + weight < B[v2]:
                B[v2] = B[v1] + weight
                f[v2] = v1
        I_marked.add(v1)
    return B


if __name__ == '__main__':
    main()
