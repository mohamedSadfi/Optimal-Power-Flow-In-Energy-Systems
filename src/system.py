import copy
import csv
import numpy as np



""" Function that create the line admittance matrix Y for the grid described by params """
def update_Ybus(params):
    Node_Num = params["Node_Num"]
    Line_Num = params["Line_Num"]
    lines = params["Network"]["Line"]

    # Initialize Ybus matrix with zeros
    Ybus = np.zeros((Node_Num, Node_Num), dtype=complex)

    for line in lines:
        start_node = line["StartNode"] - 1  # Adjusting for 0-based indexing
        end_node = line["EndNode"] - 1
        impedance = line["Impedance"]
        admittance = 1 / impedance

        # Off-diagonal elements
        Ybus[start_node, end_node] -= admittance
        Ybus[end_node, start_node] -= admittance

        # Diagonal elements
        Ybus[start_node, start_node] += admittance
        Ybus[end_node, end_node] += admittance

    return Ybus


""" Function to create a new electric grid with num_nodes nodes and num_generators generators 
    using the one defined by params as a basis """
    
def create_system(params, num_nodes, num_generators, new_system_load_P = 200, new_system_load_Q = 50):
    # Make a deep copy of the params dictionary
    new_params = copy.deepcopy(params)

    # Update the number of nodes and generators
    new_params["Node_Num"] = num_nodes
    new_params["Unit_Num"] = num_generators
    new_params["Line_Num"] = (
        num_nodes * (num_nodes - 1) // 2
    )  # Number of unique connections in a fully connected graph

    # Define identical line parameters
    lines = []
    for i in range(1, num_nodes + 1):
        for j in range(i + 1, num_nodes + 1):
            impedance = (0.01 + 0.002 * np.random.rand()) + 1j * (
                0.1 + 0.02 * np.random.rand()
            )
            lineCapacity = 100 + 100*np.random.rand()
            lines.append({"StartNode": i, "EndNode": j, "Impedance": impedance, "LineCapacity": lineCapacity,})

    new_params["Network"]["Line"] = lines

    # Update the nodes load factors (setting all to a default value, for example, 1/num_nodes)
    new_params["Network"]["Node"] = [
        {"LoadFactors": np.random.rand(), "Vmin": 0.95, "Vmax": 1.05}
        for _ in range(num_nodes)
    ]

    # Define generator parameters with tighter bounds and non-linear cost functions
    new_params["Gen"] = [
        {
            "Pmin": 0,
            "Pmax": 50 + 50 * np.random.rand(),
            "Qmin": -50,
            "Qmax": 50,
            "Cost_alpha": 0.5 + 0.5 * np.random.rand(),
            "Cost_beta": 10 + 20 * np.random.rand(),
            "Cost_gamma": 100 + 200 * np.random.rand(),
        }
        for _ in range(num_generators)
    ]

    # Update wind power output to accommodate more hours with variability
    new_params["Wind_P"] = [
        50 + 20 * np.random.rand()
        for _ in range(num_generators * new_params["Horizon"])
    ]

    # Update the WP2Node to place the wind farm at the first node
    new_params["WP2Node"] = [1] + [0] * (num_nodes - 1)

    # Calculate new Ybus matrix
    new_params["Network"]["Ybus"] = update_Ybus(new_params)
    
    new_params["SystemLoad_P"] = new_system_load_P
    new_params["SystemLoad_Q"] = new_system_load_Q

    return new_params




