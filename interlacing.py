import networkx as nx

from graphfunctions import get_adj_eigenvalues
"""
Observing the interlacing theorem in practice.
"""

def main():
    G = nx.Graph()

    n = 10
    G.add_nodes_from(range(10))
    G.add_edges_from([(0,1), (0,2), (0,9), (1,2), (1,3), (1,9), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9)])

    S = nx.circulant_graph(8, [1])
    T = nx.circulant_graph(16, [1,2])

    #print('small ',nx.adjacency_matrix(S).astype(float))
    #print('large ',nx.adjacency_matrix(T).astype(float))
    #print('goal ',nx.adjacency_matrix(G).astype(float))

    for name, graph in [("S = C_8", S), ("G (non-uniform, 10 vertices)", G), ("T = C_16(1,2)", T)]:
        #print(f"{name}\n  adjacency matrix:\n{nx.to_numpy_array(graph, dtype=int)}")
        print(f"{name} eigenvalues: {get_adj_eigenvalues(graph, 8)}\n")


main()




