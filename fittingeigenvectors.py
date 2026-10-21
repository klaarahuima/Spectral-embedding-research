import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import graphfunctions as fun

"""
Used to plot sinusoidal parameters of the eigenvector equations with respect to graph order.
I used this code to determine that the parameters followed a power-law relationship with respect to graph order.
"""
# Define the model sinusoidal function
def sinusoidal(x, A, f, phi, offset):
    # Here I am assuming the eigenvector matches a cosine function. You can change this to sine by using np.sin instead.
    return A * np.cos(2 * np.pi * f * x + phi) + offset

def plot_sinfit_and_vector(vector_coords, params):
    """
    File used to plot sinusoidal curve fits over  eigenvectors.
    Running the file will create a plot with your eigenvector overlayed by a sinusoidal function that
    best fits the data.
    y - axis : eigenvector vertex value
    x - axis: vertex index (scaled)
    """

    # y_data : coordinates of an eigenvector
    y_data = np.array(vector_coords)

    # Generate x values (assuming x values are evenly spaced from 0 to 1)
    x_data = np.linspace(0, 1, len(y_data))

    # Generate x values for plotting the estimated function
    x_plot = np.linspace(0, 1, 1000)

    # Calculate the estimated function values
    y_plot = sinusoidal(x_plot, *params)

    # Plot the data and the estimated function
    plt.plot(x_data, y_data, 'bo', label='Data')
    plt.plot(x_plot, y_plot, 'r-', label='Estimated function')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data and Estimated Sinusoidal Function')
    plt.show()


def get_parameters(orders = [200,300,400,500,600,700,800,900,1000],
                   threshold_degree :int = 72,
                   p : float = 1.,
                   q : float = .0):
    """
    Calculates the parameters of 4 sinusoidal fits: 1st quadrant / 2nd - 4th quadrant of
    2nd laplacian eigenvector / 3rd laplacian eigenvector.

    Args :

        orders is list of graphs of which you want to plot their sinusoidal parameters
        threshold_degree is max distance to 'near-by' vertices in degrees
        p = 1 probability of connecting two vertices within the threshold distance
        q = 0 probability of connecting two vertices not withing the threshold distance

        returns dictionary (v) of all parameters of sinusoidal fits to end and 3rd eigenvector
        fitted over Q1 and Q2-4 separately : see form below
    """
    # below: initializing empty dictionary for storing parameters
    # s1: 1st quadrant of second eigenvector
    # s2: 2nd-4th quadrant of 2nd eigenvector
    # t1: 1st quadrant of third eigenvector
    # t2: 2nd-4th quadrant of 3rd eigenvector
    # stored variables: amplitude, frequency, phi, offsets
    v = {
        'amps': {'s1':[], 's2':[], 't1':[], 't2':[]},
        'freqs': {'s1':[], 's2':[], 't1':[], 't2':[]},
        'phis': {'s1':[], 's2':[], 't1':[], 't2':[]},
        'offs': {'s1':[], 's2':[], 't1':[], 't2':[]}
    }
    
    threshold = np.deg2rad(threshold_degree)

    # body of code: generating graphs of all defined orders
    for n in orders:
        n_cry = n//5*2 # number of vertices in first quadrant
        # You can choose what kind of graphs to generate, uncomment which model you want to use
        # I compared random graphs and ideal circulant graphs (as defined in circulants.py, and called by cry.generate_graph(n))
        # G = cry.generate_graph(n)
        G = fun.random_graph_unit_circle(n, threshold, p, q)
        fun.add_eigenvectors(G, 0, 1)
        vectors = fun.split_vectors(G, n_cry) # getting a 4-tuple of the vectors we are studying
        for i in range(len(vectors)): # iterate over the 4 vectors
            y_data = vectors[i] # eigenvector data
            x_data = [i for i in range(len(y_data))] # creating list of indices

            # Define the sinusoidal function
            if i == 0 or i == 1: # 2nd eigenvector, 1st quadrant (i==0) or 2nd-4th quadrant (i==1)
                # in this case I assume its a cosine function
                def sinusoidal(x, A, freq, phi, offset):
                    return A * np.cos(2*np.pi * freq * x + phi) + offset
            else:
                # otherwise its the 3rd eigenvector, and I assume its a sine function
                def sinusoidal(x, A, freq, phi, offset):
                    return A * np.sin(2*np.pi * freq * x + phi) + offset
            # getting correct label based on which vector it is
            if i == 0:
                label = 's1'
                title = 'Second eigenvector, Q1'
            elif i == 1:
                label = 's2'
                title = 'Second eigenvector, Q2-Q4'
            elif i == 2:
                label = 't1'
                title = 'Third eigenvector, Q1'
            else:
                label = 't2'
                title = 'Third eigenvector, Q2-Q4'
            initial_guesses = [0.05, 0.0005, 1, 0] # guessing the 4 defining parameters
            print(f'Fitting order {n} and vector {label}')

            # Fit the sinusoidal function to the data
            # using curve_fit from scipy.optimize
            params, _ = curve_fit(sinusoidal, x_data, y_data, p0=initial_guesses, maxfev=1000000)
            A_fit, freq_fit, phi_fit, offset_fit = params # extracting parameters
            mod_p= (phi_fit % np.pi + np.pi) % np.pi # normalizing phi

            # adding the fitted parameters to the dictionary in which we store all of them
            v['amps'][label].append(A_fit)
            v['freqs'][label].append(freq_fit)
            v['phis'][label].append(mod_p)
            v['offs'][label].append(offset_fit)

            plot_sinfit_and_vector(y_data, params)
            """
            #print(f"Fitted Parameters for vector_{i}:\nAmplitude: {A_fit}\nFrequency: {freq_fit}\nPhase Shift: {phi_fit}\nOffset: {offset_fit}")

            # Here we are getting points from the fitted sinusoidal curve
            x_fit = np.linspace(min(x_data), max(x_data), 500)  # Dense x values for a smooth curve
            y_fit = sinusoidal(x_fit, A_fit, freq_fit, phi_fit, offset_fit) # Y-values

            # Here we are plotting the empirical data and fitted curve
            plt.figure(figsize=(8, 6))
            plt.scatter(x_data, y_data, label="Eigenvector coordinates", color="red", alpha=0.8, s = 5)
            plt.plot(x_fit,y_fit, label="Fitted sinusoidal curve", color="blue", alpha=0.8, linewidth=1.3)
            plt.title(title)
            plt.legend()
            plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            plt.show()
            """
            
    return v

def visualise_parameters(v, orders = [200,300,400,500,600,700,800,900,1000]):
    """
        Function to study how the parameters change wrt the order. 
        Creates a plot that visualizes each parameter ( 4 parameters for each vector = 16 in total)

        orders of graphs as list
        v dictionary containing all parameters as returned by function above
    """
    # defining 4 x 4 grid
    categories = ["amps", "freqs", "phis", "offs"]
    labels = ["s1", "s2", "t1", "t2"]

    # Create a 4x4 subplot layout
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle("Sinusoidal Fit Parameters Across Orders", fontsize=16)

    # Iterate over subcategories and labels to plot each individually
    for row, category in enumerate(categories):
        for col, label in enumerate(labels):
            ax = axes[row, col]  # Get subplot position

            ax.plot(orders, v[category][label], marker="o", linestyle="-", color="b")

            ax.set_title(f"{category.capitalize()} - {label}")
            ax.set_xlabel("Order")
            ax.set_ylabel(category.capitalize())
            ax.grid(True)

    # Adjust layout for readability
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


