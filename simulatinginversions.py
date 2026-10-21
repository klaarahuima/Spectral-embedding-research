import networkx as nx
import graphfunctions as fun
import matplotlib.pyplot as plt

"""
Used to observe the number of inversions in large graphs, specifically the relationship between
proportion of inverted vertices vs. graph order.

Running the file will do the work for you. Can edit orders of graphs you want to generate, number of simulations you want to run,
random graph parameters, and resulting plot properties.
"""

orders = [500,750,1000,1500,2000] # define different orders that you want to collect data of

# this part below is the store the data the code generates as it runs (my code, more often than not, would terminate
# before completion, resulting in all the data generated over an hour or more being lost
# will create a file in the same folder in which you are running your code
with open("simulation_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Order", "Error Proportion"])  # Column headers

# main simulation code
# p : probability two vertices within a given threshold angle are connected (1)
# q : probability any two other vertices are connected (0)
# num_simulations is number of generated graphs PER ORDER
def run_simulations(orders,p = 1, q =0, num_simulations=20):
    t = 72 # threshold angle in degrees
    avg_inversions = []
    avg_error_proportions = []
    with open("simulation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for n in orders: # iterate over every order
            errors = 0
            runs = 0
            total_error_proportions = 0
            #total_pairs = n * (n - 1) / 2
            for i in range(num_simulations): # iterate over every instance of this order
                print(f'simulating order {n} run n. {i+1}') # prints out what order we're simulating to follow progress
                G = fun.random_graph_unit_circle(n, t, p, q) # generate randomg graph according to model
                fun.add_eigenvectors(G, 0, 1)
                if nx.is_connected(G): # only count graphs that are connected (otherwise messes up results)
                    error = fun.error(G) # calculating inversions, function defined in graphfunctions.py
                    errors += error

                    runs += 1
                    writer.writerow([n, error]) # storing data in file
                    # Calculate error proportion (inversions / total possible inversions)
                    num_pairs = n * (n - 1) // 2  # Number of possible pairs of vertices
                    error_proportion = error / num_pairs if num_pairs > 0 else 0 # error proportion, this is the one that should be decreasing
                    total_error_proportions += error_proportion

                # Calculate the average values for this graph order
            avg_inversions.append(errors / runs)
            avg_error_proportions.append(total_error_proportions / runs)

    return avg_inversions, avg_error_proportions

# here we are running the actual simulation (calling the function above)
avg_inversions, avg_error_proportions = run_simulations(orders)


# Below we are creating a plot of the simulated data. Youc an of course modify the layout and look of it.

# Plotting the results
plt.figure(figsize=(8, 6))

# Plot average inversions
plt.plot(orders, avg_error_proportions, marker='o', color='r')

# Corrected title method
plt.title('Average Error Proportion (Inversions / Total Pairs)')

# Corrected axis labels
plt.xlabel('Order')
plt.ylabel('Average Error Proportion')

plt.grid(True)  # Add gridlines
plt.tight_layout()
plt.show()
