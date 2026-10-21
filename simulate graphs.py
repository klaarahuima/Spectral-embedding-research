import networkx as nx
import numpy as np
import graphfunctions as fun
from graphfunctions import get_adj_eigenvalues, get_adjacency_eigenvectors

"""
THIS IS THE FILE I GENERATE THE RANDOM GRAPHS WITH.
You can define different ways to generate random graphs, and choose which properties you want to display.
(embedding, eigenvectors, degree distribution, error, eigenvalues... etc.)

"""

# Define parameters of a random graph
order = 1000
threshold_degree = 30 #max distance to 'near-by' vertices in degrees
p = 1 #probability of connecting two vertices within the threshold distance
q = 0 #probability of connecting two vertices not withing the threshold distance
likelihood = 2
max_inversions = order*(order-1)/2



# DIFFERENT KINDS OF RANDOM GRAPHS YOU CAN GENERATE
# uncomment the one you want to generate

# BELOW: A random graph using the parameters defined above. You can change the angle distribution directly in the function call in graphfunctions.py

# G= fun.random_graph_unit_circle(order, threshold, p, q)

# BELOW: An ideal circulant graph, such that the 1st quadrant has EXACTLY twice as many vertices in 1st quadrant, and all vertices are evenly spaced
# Vertices in 1st quadrant connected to 4 nearest vertices, and the rest of the vertices are connected to 2 nearest vertices
# Used to observe behaviour of an 'ideal' version of the random graph model
# uncomment both lines below

# G = cry.generate_graph(order)
# fun.add_eigenvectors(G, 0, 1)

# BELOW: A nearest neighbours model graph, with 10 000 vertices and every vertex is connected to its 1500 nearest neighbours

# G = fun.nearest_neighbours(10000, 1500)

#BELOW: A general cycle graph

# G = nx.cycle_graph(order)
# fun.add_eigenvectors(G, a = 0, l = 1)


#s_tuple = (G.graph['second_lap'], G.graph['third_lap'])
#rads, distances = fun.eigenvector_polar_coordinates(s_tuple)
#fun.plot_distances(distances)


"""
Below you can choose what features to display.
Set the boolean to True if you want it displayed.
"""

graph_networkx = False #networkx default visualisation
graph_circle = False #vertices and edges on the unit circle visualisation (works only for graphs with vertices defined on a unit circle)

adjacency_embedding = False #embedding of 2nd and 3rd adjacency eigenvectors
laplacian_embedding = True # embedding of 2nd and 3rd laplacian eigenvectors

adj_eigenvectors = False # plot of adjacency eigenvector coordinate vs. vertex index
lap_eigenvectors = True # same for laplacian

split_vectors = False # plot of eigenvectors split into 2 plots: one for the 1st quadrant, and another for the 2nd to 4th quadrant
# interesting because the different regions of the eigenvectors exhibit different behaviour

# Uncomment below if you want to print the different properties
#print('Adjacency eigenvalues: ', G.graph['a_eigenvalues'])
#print('Laplacian eigenvalues: ', G.graph['l_eigenvalues'])
#print('Average degree: ', 2* G.size()/G.order())
#print("Average number of neighbours within threshold degree: ", order/360 * threshold_degree * 2 * p) #useful for random graphs

adj_error = False #calculates kendall distance of ordering of vertices
lap_error = False # same for Laplacian

vertex_distribution = False #histogram of how many vertices are in each quadrant
degree_distribution = False #degree histogram
plot_inversions = False # not sure...

if graph_networkx:
    fun.draw_graph(G)

if graph_circle:
    fun.plot_graph_unit_circle(G, title=f"Random Graph")

if adjacency_embedding:
    if nx.is_connected(G):
        fun.plot_unscaled_eigenvectors(G, "adj")
    else:
        print('not connected')

if laplacian_embedding:
    if nx.is_connected(G):
        print('connected')
        fun.plot_unscaled_eigenvectors(G, "lap")

if adj_eigenvectors:
    fun.plot_eigenvector(G.graph['second_adj'], G.graph['third_adj'])

if lap_eigenvectors:
    fun.plot_eigenvector(G.graph['second_lap'], G.graph['third_lap'])

if split_vectors:
    fun.plot_eigenvector(G.graph['second_lap'][:2 * order // 5], G.graph['third_lap'][:2 * order // 5])
    fun.plot_eigenvector(G.graph['second_lap'][2 * order // 5:], G.graph['third_lap'][2 * order // 5:])

if adj_error:
    err = fun.error(G, "adj")
    print("adjacency error: ",err)
    print("adjacency error portion: ", err/max_inversions)

if lap_error:
    err = fun.error(G, "lap")
    print("laplacian error: ",err)
    print("laplacian error portion: ", err/max_inversions)

if vertex_distribution:
    fun.vertex_histogram(G.graph['dictionary'])

if degree_distribution:
    avg = fun.plot_degree_distribution(G)
    print("average degree: ", avg)






