import numpy as np
import matplotlib.pyplot as plt


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


plot_nodes_vs_variables()
plot_nodes_vs_constraints()
