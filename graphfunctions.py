import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

"""
This file contains all the functions that are used to generate graphs.
You don't have to modify this code while running, the files that actually handle the simulating will call on these functions directly.
If, however, you want to modify what the functions are doing (how many eigenvectors you calculate, 
what kind of random distributions you use for generating points around the circumference of the circle, change the title of the graphs, etc.),
you should modify these functions and recompile the file your using the simulate the graphs (otherwise the edits will not be taken into account
when running). """


# BASIC GRAPH TOOLS ------------------------------------------

def draw_graph(G): #with networkx
    """
    draw_graph calls the automatic networkx graph drawing function. You can change the parameters in the function call to modify the result.

    Input: a networkx graph G
    Variables that can be changed: with_labels, node_color, edge_color, node_size (as detailed in networkx documentation)
    Output: none (draws the graph)
    """
    nx.draw(G, with_labels=False, node_color="lightblue", edge_color="gray", node_size=4)
    plt.show()


# EIGENVECTOR TOOLS -------------------------------------------

def get_eigenvectors(nx_graph):
    """
    get_eigenvectors calculates the 2nd and 3rd Laplacian eigenvector and their associated eigenvalues

    Input: a networkx graph nx_graph
    Variables that can be changed: number of eigenvectors/values computed, which eigenvector/values will be computed (smallest or largest)
    Output: a 3-tuple (2nd eigenvector, 3rd eigenvector, list of the first k eigenvalues)
    """
    A = nx.laplacian_matrix(nx_graph).astype(float)
    eigenvalues, eigenvectors = eigsh(A, k=4, which='SM')
    # above, k is the number of eigenvectors calculated
    # which ('SM') means ordering the eigenvalues by Smallest Magnitude.
    # You can change this parameter if you want the largest ones (look up scpy.sparse.linalg eigsh)
    second_eigenvector = eigenvectors[:,1]
    third_eigenvector = eigenvectors[:,2]
    return second_eigenvector, third_eigenvector, eigenvalues

def add_eigenvectors(nx_graph, a = 1, l = 1):
    """
    add_eigenvectors adds the adjacency/laplacian matrix as a graph property to a newtworkx graph

    Input: networkx graphs nx_graph, booleans a and l (signifies whether we add adjacency / laplacian eigenvectors)
    if a = 0, no adjacency eigenvectors. if a = 1, we add the adjacency eigenvectors, and similarly for l (laplacian eigenvectors)
    if you don't need the adjacency / laplacian eigenvectors, keep them as 0, otherwise you are doing alot of unnecessary computations.
    Output: nothing interesting
    """
    attrs_g = {'second_lap': [], 'third_lap': [], 'second_adj' : [], 'third_adj' : [], 'l_eigenvalues' : [], 'a_eigenvalues' : []}
    nx_graph.graph.update(attrs_g)
    # above, adding attributes to nx_graph where we can store the eigenvectors
    if l: # adding laplacian eigenvectors
        second, third, eigenvalues = get_eigenvectors(nx_graph) # calling a function I defined in this file
        nx_graph.graph['second_lap'] = second
        nx_graph.graph['third_lap'] = third
        nx_graph.graph['l_eigenvalues'] = eigenvalues
    if a: # adding adjacency eigenvectors
        news, newt, eigs = get_adjacency_eigenvectors(nx_graph) # calling a function I defined in this file
        nx_graph.graph['second_adj'] = news
        nx_graph.graph['third_adj'] = newt
        nx_graph.graph['a_eigenvalues'] = eigs
    return 1


def get_adjacency_eigenvectors(nx_graph):
    """
    get_adjacency_eigenvectors calculates the 2nd and 3rd adjacency eigenvector and their associated eigenvalues

    Input: a networkx graph nx_graph
    Variables that can be changed: number of eigenvectors/values computed, which eigenvector/values will be computed (smallest or largest)
    Output: a 3-tuple (2nd eigenvector, 3rd eigenvector, list of the first k eigenvalues)
    """
    A = nx.adjacency_matrix(nx_graph).astype(float)
    eigenvalues, eigenvectors = eigsh(A, k=4, which='LA')  # 'LA' finds largest magnitude eigenvalues
    second_eigenvector = eigenvectors[:,1]
    third_eigenvector = eigenvectors[:,2]
    return second_eigenvector, third_eigenvector, eigenvalues


def get_adj_eigenvalues(nx_graph, k:int = 10):
    """
    get adj_eigenvalues uses the same method (eigsh) as above but only returns the eigenvalues

    Input: networkx graph nx_graph
    Output: list of 10 largest eigenvalues (number can be changed)
    """
    A = nx.adjacency_matrix(nx_graph).toarray().astype(float)
    eigenvalues, _ = eigsh(A, k=k, which='LA') # k is number of eigenvalues, 'LA' means largest something..
    return eigenvalues

def get_lap_eigenvalues(nx_graph):
    """
    Same as above but for Laplacian eigenvalues
    Output: a list of 3 eigenvalues of smallest magnitude (in order)
    """
    A = nx.laplacian_matrix(nx_graph).toarray().astype(float)
    eigenvalues, _ = eigsh(A, k=3, which='SM')
    return eigenvalues


def eigenvector_polar_coordinates(eigenvector_tuple):
    """
    eigenvector_polar_coordinates transforms a linear representation of 2 eigenvectors in a 2d-plane into polar coordinates
    necessary for visualisation and computing number of inversions in the embedding

    Input: tuple of 2 eigenvectors
    Output: tuple of 2 lists: radial_angles (list of radial angles of points defined by 2nd and 3rd eigvector), distances (of the same points)
    """
    second, third = eigenvector_tuple
    n = len(second) # gets number of points
    distances = [] #initializing lists
    radial_angles = []
    for i in range(n):
        x = second[i] # getting xy-plane coordinates of every points
        y = third[i]
        theta = np.arctan2(y, x)  # Compute angle in radians
        distances.append(np.sqrt(x ** 2 + y ** 2)) # compute and append distance
        if theta < 0:  # Convert to [0, 2pi] range
            theta += 2 * np.pi
        radial_angles.append(theta) # append the calculated angle
    return radial_angles, distances

# previously cry.split_vectors(G, num_circulant)
def split_vectors(G, num_circulant):
    """
    The function splits the 2nd and 3rd laplacian eigenvector of a networkx graph G into 2 parts:
    One with vertices in 1st quadrant, and one with vertices in remaining quadrants.
    The function ignores vertices on the boundary of the 1st quadrant to minimize the deviations it has on the vertices there.

    I used this to derive the constants of the sinusoidal functions that describe the eigenvectors of the
    ideal random circulant graph model.

    Args:
        G: graph
        num_circulant: number of vertices in 1st quadrant

    Returns: 4 vectors:
            s: 2nd laplacian eigenvector, 1st quadrant
            p: 2nd laplacian eigenvector, 2nd-4th quadrant
            r: 3rd laplacian eigenvector, 1st quadrant
            q: 3rd laplacian eigenvector, 2nd-4th quadrant

    """
    s = G.graph['second_lap'][2:num_circulant-2]
    p = G.graph['second_lap'][num_circulant:]
    r = G.graph['third_lap'][2:num_circulant-2]
    q = G.graph['third_lap'][num_circulant:]
    return (s, p ,r, q)


# METRICS / ERROR FUNCTIONS -----------------------------

def count_inversions(normalized_angles):
    """
    Counts the number of inversions in a list of normalized angles.
    It compares the order of the angles with the order of vertices in the list.

    Input: list normalized_angles of radial angles (in polar coordinates) of a vector of points
    Output: tuple: count (number of inverted pairs), inverted (list of vertices that are in inverted order for at least one pair)
    """
    n = len(normalized_angles)
    sorted_indices = sorted(range(n), key=lambda i: normalized_angles[i]) # sorting angles
    inverted = []
    # Count inversions in the sorted order
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sorted_indices[i] > sorted_indices[j]:
                count += 1
                if i not in inverted:
                    inverted.append(i)
                if j not in inverted:
                    inverted.append(j)
    return count, inverted


def count_wrong_positions(radial_angles):
    """
    Count the number of inverted pairs in the circular order defined by radial_angles,
    considering both clockwise and counterclockwise directions.

    Input: list radial_angles (angle in polar coordinates of points of a vector in the original order (not sorted))
    Output: a tuple inversions, list
            inversions: number of inverted pairs
            list: list of vertices that are inverted at least once

    The function returns the inversions in the direction (clockwise / counterclockwise) with least inversions.
    """
    # Normalize angles relative to the first angle
    base_angle = radial_angles[0]
    normalized_angles = [(angle - base_angle) % (2 * np.pi) for angle in radial_angles]

    # Count inversions in both directions
    inversions_ccw, list_ccw = count_inversions(normalized_angles) # getting inversion and list counterclockwise
    reversed_angles = [(2 * np.pi - angle) % (2 * np.pi) for angle in normalized_angles] # reversing order of angle list
    inversions_cw, list_cw = count_inversions(reversed_angles) #getting inversion and list clockwise
    #Returning direction with least inversions
    if inversions_cw < inversions_ccw:
        return inversions_cw, list_cw
    else:
        return inversions_ccw, list_ccw

def error(nx_graph, type = "lap"):
    """
    error calculates how many inversions the 2d embedding of a networkx graph has by calling functions I defined above

    The code might be faulty

    Input: networkx graph nx_graph, eigenvector type "lap" or "adj"
            if type is not specificed, it assumes laplacian eigenvectors
    Output: number of inversions in the embedding of the 2nd and 3rd eigenvectors
    """
    if type == "adj": # if type = "adj", we use adjacency eigenvectors
        second, third = nx_graph.graph['second_adj'], nx_graph.graph['third_adj']
    elif type == "lap": # else laplacian
        second, third = nx_graph.graph['second_lap'], nx_graph.graph['third_lap']
    else: # else tells you your type is not valid
        print('Invalid type for error')
        return
    rads, distances = eigenvector_polar_coordinates((second, third)) # getting polar coordinates
    err, _ = count_wrong_positions(rads) # calculates inversions
    return err



# GENERATING VERTICES TOOL FUNCTIONS --------------------------

def angles_to_dictionary(angles, order):
    """
    Takes two arguments: a list of angles (angular coordinates of vertices) and the order of the graph.
    Returns: A dictionary where every angle is assigned a position (ordering).
    A helper function, not very interesting for graph study purposes.

    Input: list of angles, number of angles
    Output: dictionary with keys: indices and values: vertex_angles
    """
    unsorted = [abs(x) for x in angles]
    vertex_angles = sorted(unsorted) # sorts angles
    keys = [i for i in range(order)] # index list
    dictionary = dict(zip(keys, vertex_angles)) # combine angles and indices into a dictionary
    return dictionary


def uniform_angles(n):
    """
    Generates n vertices uniformly around the unit circle. (Used to generate vertices uniformly around a circle.)

    Input: number of vertices n
    Output: dictionary with keys: indices and values: vertex_angles
    """
    unsorted = np.random.uniform(0, 2 * np.pi, n)
    return angles_to_dictionary(unsorted, n)

def non_uniform_angles(n):
    """
    Generates n vertices non-uniformly around the unit circle.
    In this model, there are approximately twice as many vertices in the first quadrant than in the other quadrants.
    Used for simulating graphs with non-uniform distributions.

    Input: number of vertices n
    Output: dictionary with keys: indices and values: vertex_angles
    """
    raw = np.random.uniform(-np.pi/2, 2*np.pi, n)
    # generates twice as many angles in q1 by sampling from -pi/2 to 2pi, and then adding the vertices in range (-pi/2, 0)
    # to the range (0, pi/2)
    return angles_to_dictionary(raw, n)


def any_angle(likelihood, n):
    """
    Used to generate angles around a circle, such that the proportion of vertices in the first quadrant (compared to the rest of the graph)
    can be defined separately (not necessarily 2, like above).

    Input:
        likelihood: how much more likely a vertex is to be in the first quadrant
        n: number of vertices to generate

    Output: dictionary with keys: indices and values: vertex_angles
    """
    angles = []
    p = 1 / (likelihood + 3)
    range_prob = likelihood * p
    for _ in range(n):
        if np.random.rand() < range_prob: # generate random number from (0,1) and check if less than range_prob
            # this is a common way to using probabilities in code. On average, a portion range_prob will be sampled from this loop.
            ang = np.random.uniform(0, np.pi/2)
            # if less than range_prob, we sample from (0, pi/2)
        else:
            ang = np.random.uniform(np.pi/2, 2*np.pi)
            # otherwise we sample an angle from the remaining 3 quadrants.
        angles.append(ang)
    return angles_to_dictionary(angles, n)


def only_quartile(n):
    """
    Samples angles on from first quartile.

    Input: n: number of angles to generate
    Output: dictionary like all the ones above
    """
    raw = np.random.uniform(0,np.pi/2, n)
    return angles_to_dictionary(raw, n)

def non_uniform_angles_2(n):
    """
    Generates n vertices non-uniformly around the unit circle.
    There are approximately twice as many vertices in the first quadrant than in the other quadrants.

    This is just another version of non_uniform_angles.
    """
    angles = np.random.uniform(-3*np.pi/2, 2*np.pi, n)
    for i in range(len(angles)):
        if angles[i] < 0:
            if angles[i] >= -np.pi/2:  # Between -pi and 0
                angles[i] += np.pi / 2
            elif angles[i] >= -np.pi:  # Between -2pi and -pi
                angles[i] += np.pi
            else:  # Between -3pi/2 and -2pi
                angles[i] += 3*np.pi / 2
    return angles_to_dictionary(angles, n)



# GENERATE RANDOM GRAPHS ----------------------------

def random_graph_unit_circle(order = 600, threshold = 25, p = 1, q = 0, likelihood = 2):
    """
    MOST IMPORTANT FUNCTION.
    Used for simulating random graph according to our model.

    Input:
        order: number of vertices in graph
        threshold: threshold degree for connecting 2 vertices in degrees (not radians! I made this mistake so many times)
        p: probability that two vertices within the threshold distance are connected
        q: probability that two vertices not within the threshold distance are connected
        likelihood: proportion of vertices in first quadrant (integer value, for example 3x as many vertices in q1 than other quadrants)
                - only referenced if you generate vertices with the any_angles function

    Returns: a networkx graph according to the defined properties
    """

    # Below, we generate angles for the vertices
    # Uncomment which kind of distribution of angles you want to use

    angle_dict = uniform_angles(order) # uniform distribution of angles
    #angle_dict = non_uniform_angles(order) # twice as many vertices in q1
    #angle_dict = any_angle(likelihood, order) # any proportion
    threshold = np.deg2rad(threshold)

    G = nx.Graph(dictionary = angle_dict) # creating angle dictionary
    for i in angle_dict.keys():
        G.add_node(i, angle=angle_dict[i])  # Add nodes with their angle as an attribute

    # Connect edges based on the distance and probabilities
    for i in range(order):
        for j in range(i + 1, order):
            # Calculate the angular distance between nodes i and j
            abs_diff = abs(G.nodes[i]["angle"] - G.nodes[j]["angle"])
            angle_diff = min(abs_diff, 2 * np.pi - abs_diff)  # Circular distance

            # Decide connection probability
            prob = p if angle_diff < threshold else q

            # Add edge with the computed probability
            if np.random.rand() < prob:
                G.add_edge(i, j)

    return G

def nearest_neighbours(order, neighbours):
    """
    Creates random graph like above but instead of using a threshold to connect vertices, every vertex is connected to its nearest neighbours (number defined).

    Input: order of graph
            neighbours: number of nearest neighbours each vertex is connected to ( can be unevenly on either side)
    """
    # Generate angles for each vertex
    angle_dict = non_uniform_angles(order) # change this function to modify the angle distribution
    # Initialize the graph with nodes and their angles
    G = nx.Graph(dictionary = angle_dict)
    for i in angle_dict.keys():
        G.add_node(i, angle=angle_dict[i])
    # Connect each node to its closest neighbors by angular distance
    for i in angle_dict.keys():
        distances = []
        for j in angle_dict.keys():
            if i != j:
                abs_diff = abs(angle_dict[i] - angle_dict[j])
                angle_diff = min(abs_diff, 2 * np.pi - abs_diff)  # Circular distance
                distances.append((j, angle_diff))
        # Sort neighbors by angular distance
        distances.sort(key=lambda x: x[1])
        # Connect to the closest neighbors
        for neighbor, _ in distances[:neighbours]:
            G.add_edge(i, neighbor)

    return G

# previously cry.generate_graph(n)
def ideal_circulant_denseQ1(n):
    """
    Function used to generate ideal circulant graphs with exactly twice as many vertices in 1st quadrant (not random).

    Args:
        n: order of generated graph

    Returns: networkx graph of the model described above
    """
    G = nx.Graph() # create empty graph
    num_circulant = n//5*2 # number of vertices in 1st quadrant
    k = 2  # Number of closest neighbors on each side
    G.add_nodes_from([i for i in range(n)])
    # below: adding edges at the boundary between 1st quadrant and rest of graph
    G.add_edge(0, 1)
    G.add_edge(0,2)
    G.add_edge(1,3)
    G.add_edge(1, 2)
    for i in range(2,num_circulant-2): # adding all the other edges in 1st quadrant
        G.add_edge(i, (i + 1))  # Add edge to the right
        G.add_edge(i, (i - 1))  # Add edge to the left
        G.add_edge(i, (i - 2))  # Add edge to the left
        G.add_edge(i, (i + 2))  # Add edge to the left
    G.add_edge(num_circulant-2,num_circulant-1)

    for i in range(num_circulant,n-1): # adding edges to rest of graph
        G.add_edge(i, (i + 1))  # Add edge to the next vertex
        G.add_edge(i, (i - 1)) # Add edge to the previous vertex
    G.add_edge(n-1,0)
    fun.add_eigenvectors(G, 0, 1)

    return G

# previously rem.simulate(n,d,p)
def removing_random_edges(n: int = 1000, d: int = 20, p:float = 0.6):
    """
    Function defined to generate graphs in which we remove edges from perfect circulant graphs (like in the Richter Rocha paper).

    Defining parameters of graph:
    d = 20  number of nearest neighbours every vertex is connected to (per side)
    n = 1000 order of graph
    p = 0.6 probability two vertices that are neighbours in original circulant graph will be connected in derived random graph
    """
    G = nx.circulant_graph(n, [i for i in range(d)]) # standard networkx function for generating circulant graph
    print(G)
    H = G.copy() # creating copy of graph
    for u,v in G.edges: # iterating over every edge in graph
        if np.random.uniform(0, 1) < 1 - p: # removing edges uniformly with probability 1 -p
            H.remove_edge(u,v)
    print(H)
    return H # return final random graph


# PLOTTING ------------------------------

def plot_graph_unit_circle(nx_graph, title="Random Graph on Unit Circle"):
    """
    Plots graph with vertices on the unit circle. Used for graphs of low order to visualise the edge connections along the circle.
    """
    # Map vertices to 2D coordinates on the unit circle
    positions = {i: (np.cos(angle), np.sin(angle)) for i, angle in nx_graph.graph['dictionary'].items()}

    # Plot the graph
    plt.figure(figsize=(8, 8))
    nx.draw(
        nx_graph,
        pos=positions,
        with_labels=False,
        node_size=3,
        node_color="blue",
        edge_color="lightgray"
    )
    plt.title(title)
    plt.axis("equal")
    plt.show()

def plot_degree_distribution(nx_graph):
    """
    This was copied from one of the Applied Graph Theory workbooks. Plots the degree distribution of vertices.
    Input: nx_graph
    Output: average_degree (also creates the plot, of course).
    """
    average_degree = sum(degree for node, degree in nx_graph.degree()) / nx_graph.number_of_nodes()
    degree_sequence = sorted((degree for node, degree in nx_graph.degree()), reverse=True)
    min_degree = min(degree_sequence)
    max_degree = max(degree_sequence)
    degree_range = max_degree - min_degree + 1
    degrees = range(min_degree, max_degree + 1)  # create a list of numbers from min degree through max degree

    deg_distribution = np.zeros(degree_range)  # initiate all counts equal to zero
    for v in nx_graph:
        d = nx_graph.degree(v)
        deg_distribution[d - min_degree] += 1  # increases the relevant counter by 1

    fig, ax = plt.subplots(figsize=(8, 6))  # Standard figure and axes creation
    ax.bar(degrees, deg_distribution, width=0.8, color='lightblue', edgecolor='none')  # Bar chart with better aesthetics
    ax.set_xticks(degrees)  # Set x tick marks to the degree values
    plt.xticks(fontsize=10)
    ax.set_xlabel("Degree", fontsize=12)  # Label for the x-axis
    ax.set_ylabel("Frequency", fontsize=12)  # Label for the y-axis
    ax.set_title("Degree Distribution", fontsize=14)  # Add a title to the plot
    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()

    return average_degree

def vertex_histogram(dictionary):
    """
    Plots a histogram visualising how many vertices are in each quadrant of the circular graph.
    Mostly used to check my random distribution functions for defining the set of vertices was working correctly.
    Not very interesting otherwise.

    Input: dictionary, like the ones returned by angles_to_dictionary
    Output: shows a histogram plot visualising how many vertices are in each quadrant
    """
    quadrant_counts = [0, 0, 0, 0]  # Counts for each quadrant
    #counting quadrant elements
    for angle in dictionary.values():
        if 0 <= angle < np.pi / 2:
            quadrant_counts[0] += 1
        elif np.pi / 2 <= angle < np.pi:
            quadrant_counts[1] += 1
        elif np.pi <= angle < 3 * np.pi / 2:
            quadrant_counts[2] += 1
        else:
            quadrant_counts[3] += 1
    labels = ['Quadrant 1', 'Quadrant 2', 'Quadrant 3', 'Quadrant 4']
    plt.bar(labels, quadrant_counts, color='skyblue')
    plt.title('Distribution of Vertices in Different Quadrants')
    plt.xlabel('Quadrant')
    plt.ylabel('Number of Vertices')
    plt.show()

def plot_unscaled_coordinates(coordinates):
    """
    Plots 2 vectors (with x and y coordinates, respectively) in the 2d-plane.
    Used to visualise the embedding of the 2nd and 3rd eigenvectors (or any 2 eigenvectors, for that matter.)

    Input: 2 vectors of coordinates (can input eigenvectors directly)
    Output: draws the emebdding
    """
    x, y = coordinates
    plt.figure(figsize=(8, 8))  # Ensure square figure

    # this part below is for labelling the vertices in the plot, if interested
    # I used it for particularly large graphs to track, for example, every 1000th vertex (to see where the quadrant fall)
    # uncomment to use :)
    """
    for i, (xi, yi) in enumerate(zip(x, y), start=1): # Start labeling from 1
        if (i == 1) or (i % 1000 == 0): # i % 1000 == 0 says we visualize every 1000th vertex. You can modify this by changing the value after %.
            plt.text(xi, yi, str(i), fontsize=10, ha='right', va='bottom', color='black') # choosing text display options (look up plt.text if needed)
    """

    # Add labels and title
    plt.title("Model embedding", fontsize = 19) # can edit properties of title
    split_idx = int(len(x) * 2 / 5) # splitting points into 1st quadrant (2/5 of all vertices) and the rest of the graph

    # Scatter first 2/5 of the points in red
    plt.scatter(x[:split_idx], y[:split_idx], s=10, color='red', label="First quadrant")

    # Scatter the remaining points in blue
    plt.scatter(x[split_idx:], y[split_idx:], s=10, color='blue', label="Remaining vertices")

    x_lim = max(abs(min(x)), abs(max(x)))  # Maximum absolute x value
    y_lim = max(abs(min(y)), abs(max(y)))  # Maximum absolute y value

    # Ensure equal aspect ratio and scaling
    plt.axis('equal')  # Better than 'equal' for auto-adjusting limits
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5) # parameters of grid
    plt.legend(fontsize = 14, loc = "upper left") # legend
    plt.scatter(0, 0, color="black", label="(0, 0)", zorder=0.4) # adding origin to plot for analysis


def plot_eigenvector(second, third):
    """
    Creates 2 plots (one for 2nd eigvec, one for 3rd eigvec)
        x-axis: vertex index
        y-axis: coordinate of vertex in the second/third eigenvector

    Input: 2 eigenvectors
    Output: shows the plots as described, one at a time
    """
    indices = [i for i in range(len(second))] # makes list of vertex indices (0 to n)
    x = indices

    # 1st figure: 2nd eigenvector
    plt.figure(figsize=(8, 6))
    plt.title("Non-uniform random graph, second eigenvector")
    plt.plot(x, second, color='b', marker='o', markersize=3, linestyle='-') # the points are connected with lines. You can change display parameteres here.
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5) # grid properties

    plt.tight_layout()
    plt.show()

    #2nd figure: 3rd eigenvector. Same functions as above.
    plt.figure(figsize=(8, 6))
    plt.title("Non-uniform random graph, third eigenvector")
    plt.plot(x, third, color='r', marker='o', markersize=3, linestyle='-')
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_unscaled_eigenvectors(nx_graph, type = "lap"):
    """
    Directly plots the 2d-embedding of a networkx graph according to laplacian or adjacency eigenvalues.
    (Combines all the functions I defined above!)

    Input: a networkx graph nx_graph; an eigenvector type "lap" or "adj"
    Output: plots the embedding of the eigenvectors
    """
    if type == "adj":
        vectors = (nx_graph.graph['second_adj'], nx_graph.graph['third_adj'])
    elif type == "lap":
        vectors = (nx_graph.graph['second_lap'], nx_graph.graph['third_lap'])
    else:
        print('Invalid type for plot_unscaled_eigenvectors')
        return
    plot_unscaled_coordinates(vectors)

def plot_distances(distances):
    """
    plot_distances creates a plot of the distance of points from the origin

    Input: list of distances of vertices (not necessarily in increasing order)
    Output: creates plot of graph such that
                x-axis: vertex index
                y-axis: distance from origin
    """
    indices = np.arange(len(distances))

    plt.figure(figsize=(8, 5)) # figure size
    plt.plot(indices, distances, marker='o', linestyle='-', label="Distances") # creates plot
    plt.title("Distances from Origin")
    plt.xlabel("Index")
    plt.ylabel("Distance")
    plt.ylim(0, max(distances) * 1.1)  # Ensures the y-axis starts from 0
    plt.grid(True) # can change grid properties here
    plt.legend()
    plt.show()



"""

Old version of a function used to calculate how many inversions were in each quadrant of the graph. Not sure if it works anymore.
Feel free to use as a starting point if needed.

def plot_inversions(nx_graph):
    second, third = get_eigenvectors(nx_graph)
    rads, _ = eigenvector_radial_angles((second, third))
    _, inversions = count_wrong_positions(rads)
    order = nx_graph.order()
    n = len(inversions)
    q = order/5
    bins = [0, q, 2*q, 3*q, 4*q, order]  # Adjust the bins as needed

    # Plot the histogram
    plt.hist(inversions, bins=bins, edgecolor='black', alpha=0.7)

    # Add labels and title
    plt.xlabel('Ranges')
    plt.ylabel('Count')
    plt.title('Number Distribution Across Ranges')

    # Show the plot
    plt.show()
"""






