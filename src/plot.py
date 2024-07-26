import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from params import params
import pandas as pd
# from scipy.interpolate import make_interp_spline
# from scipy.interpolate import griddata





def plot_nodes_vs_variables():
    # Define the number of nodes and the corresponding number of variables
    nodes = np.arange(3, 10001)
    H = 1  # Assuming a single time period (horizon)

    # Calculate the total number of variables
    total_variables = 5 * nodes + nodes * H + nodes * (nodes - 1) + nodes * (nodes - 1)

    # Plot the number of nodes vs the total number of variables
    plt.figure(figsize=(10, 6))
    plt.plot(nodes, total_variables, label='Total Variables')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Total Number of Variables')
    plt.title('Number of Nodes vs Total Number of Variables')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Use a logarithmic scale for better visualization
    plt.show()

def make_interpolation(Pgen, target_length):

    # Number of elements to interpolate
    num_original = len(Pgen)
    num_to_interpolate = target_length - num_original

    # Calculate how many values to add between each pair
    values_per_gap = num_to_interpolate // (num_original - 1)
    extra_values = num_to_interpolate % (num_original - 1)

    # Interpolation
    interpolated_array = []
    for i in range(num_original - 1):
        interpolated_array.append(Pgen[i])
        # Number of values to interpolate for this gap
        num_values = values_per_gap + 1 if i < extra_values else values_per_gap
        interpolated_values = np.linspace(Pgen[i], Pgen[i + 1], num=num_values + 2)[1:-1]
        interpolated_array.extend(interpolated_values)
    interpolated_array.append(Pgen[-1])

    # Convert to numpy array
    interpolated_array = np.array(interpolated_array)

    return interpolated_array


def evaluate_objective_function(Pgen):
    return sum(
    params["Gen"][i]["Cost_alpha"] * Pgen[i] ** 2
    + params["Gen"][i]["Cost_beta"] * Pgen[i]
    + params["Gen"][i]["Cost_gamma"]
    for i in range(2))


def plot_errors(errors):
    # length of numpy arrray
    length = len(errors)

    iterations = np.arange(1, length + 1)

    # Errors_Spline = make_interp_spline(iterations, errors)
    # iterations = np.linspace(1, length - 1, 1000)

    # errors_smooth = Errors_Spline(iterations)

    # Plot the errors on a separate graph
    fig2 = plt.figure(figsize=(12, 4))
    ax2 = fig2.add_subplot(111)
    ax2.plot(iterations, errors, color="LIME", label="Infeasibility (inf_pr)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Error (inf_pr)")
    ax2.set_title("Error vs Iteration")
    ax2.legend()
    ax2.grid(True)

    plt.show()


def plot_3d_errors(Pgen1, Pgen2, errors):
    # Define the range of Pgen values for each generator
    Pgen1_range = np.linspace(2*min(Pgen1), 2*max(Pgen1), 100)
    Pgen2_range = np.linspace(2*min(Pgen2), 2*max(Pgen2), 100)

    # Create a meshgrid for Pgen1 and Pgen2
    pgen1_ax, pgen2_ax = np.meshgrid(Pgen1_range, Pgen2_range)

    Pgen1_interpolated = make_interpolation(Pgen1, 10000)
    Pgen2_interpolated = make_interpolation(Pgen2, 10000)
    errors_interpolated = make_interpolation(errors, 10000)

    # Interpolate errors to the grid using griddata
    Z = griddata(
         (Pgen1_interpolated, Pgen2_interpolated), errors_interpolated, (pgen1_ax, pgen2_ax), method="cubic"
     )
    print(np.size(Z))

    # Create the 3D plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    errors_interpolated = errors_interpolated.reshape((100,100))

    # Plot the surface
    surf = ax.plot_surface(
        pgen1_ax, pgen2_ax, errors_interpolated, alpha=0.8
    )



    # Set labels and title
    ax.set_xlabel("Pgen1 (MW)")
    ax.set_ylabel("Pgen2 (MW)")
    ax.set_zlabel("Error Value")
    ax.set_title("3D Plot of Errors with Pgen1 and Pgen2")

    plt.show()


def plot_pgen(Pgen1, Pgen2):
    # length of numpy arrray
    length = len(Pgen1)

    iterations = np.arange(1, length + 1)

    # Pgen1_Spline = make_interp_spline(iterations, Pgen1)
    # Pgen2_Spline = make_interp_spline(iterations, Pgen2)

    # iterations = np.linspace(1, length - 1, 1000)

    # Pgen1_smooth = Pgen1_Spline(iterations)
    # Pgen2_smooth = Pgen2_Spline(iterations)

    # Plot the errors on a separate graph
    fig2 = plt.figure(figsize=(12, 4))
    ax2 = fig2.add_subplot(111)
    ax2.plot(iterations, Pgen1, color="LIME", label="Pgen1")
    ax2.plot(iterations, Pgen2, color="r", label="Pgen2", linestyle='--')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("MW")
    ax2.set_title("Pgen1, Pgen2 vs iterations")
    ax2.grid(True)
    ax2.legend()

    plt.show()


def plot_objective(Pgen1, Pgen2):
    fig, ax = plt.subplots()

    Pgen1new = make_interpolation(Pgen1, 100)
    Pgen2new = make_interpolation(Pgen2, 100)

    pgen1_ax, pgen2_ax = np.meshgrid(
        np.linspace(2*min(Pgen1new), 2*max(Pgen1new), 100),
        np.linspace(2*min(Pgen2new), 2*max(Pgen2new), 100)
    )

    Pgen = (pgen1_ax, pgen2_ax)

    Z = evaluate_objective_function(Pgen)

    # Generate the contour plot
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(pgen1_ax, pgen2_ax, Z, levels=50, cmap='viridis')

    # Add a colorbar
    cbar = plt.colorbar(contour)
    cbar.set_label("Objective Function Value")

    # Plot the iteration points on top of the contour plot
    ax.plot(pgen1, pgen2, 'ro-', label='Iterations')

    # Set labels and title
    ax.set_xlabel("Pgen1 (MW)")
    ax.set_ylabel("Pgen2 (MW)")
    ax.set_title("Contour Plot of the Objective Function with Iterations")

    minima = np.array([Pgen1[-1], Pgen2[-1]])

    ax.plot(minima[0], minima[1], "go", markersize=14, lw=0, label="Optimal Solution")

    # Add vertical and horizontal dotted lines
    ax.axhline(y=minima[1], xmin=0, xmax=0.5, color="g", linestyle="--", linewidth=1)
    ax.axvline(x=minima[0], ymin=0, ymax=0.5, color="g", linestyle="--", linewidth=1)

    # Compute the cost function at the optimal solution
    optimal_cost = evaluate_objective_function((minima[0], minima[1]))

    # Annotate the cost function value at the optimal solution
    ax.annotate(
        f"Cost: {optimal_cost:.2f}",
        xy=(minima[0], minima[1]),
        xytext=(minima[0] + 5, minima[1] + 5),
        arrowprops=dict(facecolor="green", shrink=0.05),
        fontsize=12,
        color="black",
    )

    # Annotate the Pgen1 and Pgen2 values at the intersections
    ax.annotate(
        f"Pgen1: {minima[0]:.2f} MW",
        xy=(minima[0], ax.get_ylim()[0]),
        xytext=(minima[0] + 5, ax.get_ylim()[0] + 5),
        fontsize=12,
        color="black",
    )
    ax.annotate(
        f"Pgen2: {minima[1]:.2f} MW",
        xy=(ax.get_xlim()[0], minima[1]),
        xytext=(ax.get_xlim()[0] + 5, minima[1] + 5),
        fontsize=12,
        color="black",
    )

    ax.legend()
    ax.legend(loc="best", numpoints=1)


with open("iterations.csv", "r") as file:
    lines = file.readlines()

# Ensure the header is properly formatted and split into columns
header = [
    "iter",
    "objective",
    "inf_pr",
    "inf_du",
    "lg(mu)",
    "||d||",
    "lg(rg)",
    "alpha_du",
    "alpha_pr",
     "ls",
     "bruh"
]
data_lines = [line.strip().split() for line in lines[0:]]

# Create a DataFrame from the cleaned lines
df = pd.DataFrame(data_lines, columns=header)

# Convert the 'objective' column to float
df["objective"] = df["objective"].astype(float)
objective_values = df["objective"].to_numpy()

errors = df["inf_pr"].astype(float).to_numpy()


pgen1 = []
pgen2 = []


# Open the file and read the values
with open("pgen_values.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        # Split the string and keep the value after '='
        value = float(row[0].split("=")[1])

        # Determine if the index is impair or pair
        if int(row[0].split("[")[1].split("]")[0]) % 2 != 0:
            pgen1.append(value)
        else:
            pgen2.append(value)

pgen1 = np.array(pgen1)
pgen2 = np.array(pgen2)


# Define the range of Pgen values for each generator
Pgen0_range = np.linspace(-100, 100, 100)
Pgen1_range = np.linspace(-100, 100, 100)

# Create a meshgrid for Pgen0 and Pgen1
Pgen0, Pgen1 = np.meshgrid(Pgen0_range, Pgen1_range)
Pgen = (Pgen0, Pgen1)


# Compute the function values
Z = sum(
    params["Gen"][i]["Cost_alpha"] * Pgen[i] ** 2
    + params["Gen"][i]["Cost_beta"] * Pgen[i]
    + params["Gen"][i]["Cost_gamma"]
    for i in range(2)
)


# Plot the function
fig = plt.figure(figsize=(12, 12))  # Increase figure size
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

# 3D plot
ax1 = fig.add_subplot(gs[0, 0], projection="3d")
ax1.plot_surface(Pgen0, Pgen1, Z, cmap="Greens", alpha=0.8)

# Plot objective_values
ax1.plot(
    pgen1,
    pgen2,
    objective_values,
    color="r",
    label="Objective Values",
)

# Set labels for the 3D plot
ax1.set_xlabel("Pgen0")
ax1.set_ylabel("Pgen1")
ax1.set_zlabel("Objective Function Value")
ax1.set_title("Objective Function Surface")


plot_errors(errors)
# plot_objective(pgen1, pgen2)
plot_pgen(pgen1, pgen2)



plt.show()
