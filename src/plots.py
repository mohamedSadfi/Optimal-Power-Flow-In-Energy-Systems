""" All functions used for plotting are defined here. """


import numpy as np
import matplotlib.pyplot as plt
from params import params



# Function to calculate the number of equality and inequality constraints
def count_constraints(nodes, horizon):
    equality_constraints = (
        2 * nodes * horizon  # Node Balance (P and Q)
        + 2 * nodes * (nodes - 1) * horizon  # Power Flow (P and Q)
    )
    inequality_constraints = (
        2 * nodes * horizon  # Pmax and Pmin
        + 2 * nodes * horizon  # Qmax and Qmin
        + horizon  # WPmax
        + nodes * (nodes - 1) * horizon  # Line Capacity
        + horizon  # Wind Power Generation
        + 2 * nodes * horizon  # Voltage Limits (lower and upper)
        + 2 * nodes * horizon  # Angle Limits (lower and upper)
        + 2 * horizon  # Slack Bus Angle and Voltage
    )
    return equality_constraints, inequality_constraints


def plot_nodes_vs_variables():
    # Define the number of nodes and the corresponding number of variables
    nodes = np.arange(3, 5000)
    H = 1  # Assuming a single time period (horizon)

    # Calculate the total number of variables
    total_variables = 5 * nodes + nodes * H + nodes * (nodes - 1) + nodes * (nodes - 1)

    # Plot the number of nodes vs the total number of variables
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, total_variables, label="Total Variables")
    plt.xlabel("Network Size (~Number of Nodes)")
    plt.ylabel("Total Number of Variables")
    plt.legend()
    plt.grid(True)
    # plt.yscale("log")  # Use a logarithmic scale for better visualization
    plt.show()


def plot_nodes_vs_constraints():
    # Node counts for the plot
    node_counts = [3, 10, 50, 100, 500, 1000, 2000, 5000]
    horizon = 1  # Single time period for simplicity

    # Calculate the number of constraints for each node count
    equality_constraints = []
    inequality_constraints = []
    for nodes in node_counts:
        eq, ineq = count_constraints(nodes, horizon)
        equality_constraints.append(eq)
        inequality_constraints.append(ineq)

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.4
    index = np.arange(len(node_counts))

    bar1 = plt.bar(index, equality_constraints, bar_width, label='Equality Constraints')
    bar2 = plt.bar(index + bar_width, inequality_constraints, bar_width, label='Inequality Constraints')

    plt.xlabel('Network Size (~Number of Nodes)')
    plt.ylabel('Number of Constraints')
    plt.xticks(index + bar_width / 2, node_counts)
    # plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    


def evaluate_objective_function(Pgen):
    return sum(
        params["Gen"][i]["Cost_alpha"] * Pgen[i] ** 2
        + params["Gen"][i]["Cost_beta"] * Pgen[i]
        + params["Gen"][i]["Cost_gamma"]
        for i in range(2)
    )

def plot_errors(errors):
    length = len(errors)
    iterations = np.arange(1, length + 1)

    fig2 = plt.figure(figsize=(12, 4))
    ax2 = fig2.add_subplot(111)
    ax2.scatter(iterations, errors, color="black", label="Infeasibility (inf_pr)", marker='o', s=7)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Error (inf_pr)")
    ax2.set_title("Error vs Iteration")

    # Customize x-axis labels
    ax2.tick_params(axis='x', labelsize=5, labelcolor='black')

    ax2.legend()
    ax2.grid(True)

    plt.show()


def plot_pgen(Pgen1, Pgen2):
    length = len(Pgen1)
    iterations = np.arange(1, length + 1)

    fig2 = plt.figure(figsize=(12, 4))
    ax2 = fig2.add_subplot(111)
    
    ax2.scatter(iterations, Pgen1, color="blue", label="Pgen1", marker='o', s=7)
    ax2.scatter(iterations, Pgen2, color="orange", label="Pgen2", marker='o', s=7)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("MW")
    ax2.set_title("Pgen1, Pgen2 vs iterations")
    ax2.grid(True)
    ax2.legend()

    plt.show()
    

# Function to plot the variable. Extract the values of the desired variable
# in the optimization.py file, then read the values and save it in a variable in main_plots.py with the
# save_variable_to_array function from helper.py and finally plot it with the function defined below

def plot_variable(p):
    length = len(p)
    iterations = np.arange(1, length + 1)

    fig2 = plt.figure(figsize=(12, 4))
    ax2 = fig2.add_subplot(111)
    
    ax2.scatter(iterations, p, color="blue", label="P_wp", marker='o', s=7)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("MW")
    ax2.set_title("Wind active power generation vs iterations")
    ax2.grid(True)
    ax2.legend()

    plt.show()


# Function to plot the objective function values as a function of Pgen1 and Pgen2 accros iterations 

def plot_isocost_lines(Pgen1, Pgen2):
    Pgen1_range = np.linspace(-10, 120, 100)
    Pgen2_range = np.linspace(-10, 80, 100)

    Pgen0, Pgen1_mesh = np.meshgrid(Pgen1_range, Pgen2_range)
    Pgen = (Pgen0, Pgen1_mesh)

    Z = sum(
        params["Gen"][i]["Cost_alpha"] * Pgen[i] ** 2
        + params["Gen"][i]["Cost_beta"] * Pgen[i]
        + params["Gen"][i]["Cost_gamma"]
        for i in range(2)
    )

    fig, ax = plt.subplots(figsize=(12, 12))
    contourf = ax.contourf(Pgen0, Pgen1_mesh, Z, levels=100, cmap='Greens')

    cbar = plt.colorbar(contourf)
    cbar.set_label("Objective Function Value")

    num_iterations = len(Pgen1)
    colors = plt.cm.Reds(np.linspace(0.3, 1, num_iterations))  # Create a colormap from light red to dark red
    marker_sizes = np.linspace(3, 7, num_iterations)  # Create an array of marker sizes from 3 to 7

    for i in range(num_iterations):
        ax.plot(Pgen1[i], Pgen2[i], 'o', color=colors[i], markersize=marker_sizes[i])

    ax.plot([], [], 'o', color='red', label='Iterations')
    ax.set_xlabel("Pgen1 (MW)")
    ax.set_ylabel("Pgen2 (MW)")
    ax.set_title(f"Contour Plot of the Objective Function ({num_iterations} Iterations)")

    minima = np.round([Pgen1[-1], Pgen2[-1]], 2)

    ax.plot(minima[0], minima[1], 'x', markersize=9, lw=0, label="Optimal Solution",color="black")

    ax.axvline(x=minima[0], ymin=0, ymax=0.2, color="blue", linestyle=":", linewidth=1,label = f"P_gen1: {minima[0]} MW")
    ax.axhline(y=minima[1], xmin=0, xmax=0.2, color="orange", linestyle=":", linewidth=1,label = f"P_gen2: {minima[1]} MW")

    optimal_cost = evaluate_objective_function((minima[0], minima[1]))

    ax.annotate(
        f"Cost: {optimal_cost:.2f}",
        xy=(minima[0], minima[1]),
        xytext=(minima[0] + 5, minima[1] + 5),
        arrowprops=dict(facecolor="green", shrink=0.05),
        fontsize=12,
        color="black",
    )
    

    '''ax.annotate(
        f"Pgen0: {minima[0]:.2f} MW",
        xy=(minima[0], ax.get_ylim()[0]),
        xytext=(minima[0] + 5, ax.get_ylim()[0] + 5),
        fontsize=12,
        color="black",
    )
    ax.annotate(
        f"Pgen1: {minima[1]:.2f} MW",
        xy=(ax.get_xlim()[0], minima[1]),
        xytext=(ax.get_xlim()[0] + 5, minima[1] + 5),
        fontsize=12,
        color="black",
    )'''

    ax.legend()
    ax.legend(loc="best", numpoints=1)
    