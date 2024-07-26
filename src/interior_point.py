import copy
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import re
from params import params
from system import update_Ybus, create_system

# Check if ipopt is available
if not SolverFactory("ipopt").available():
    print("Solver 'ipopt' is not available. Please check the installation.")
else:
    print("Solver 'ipopt' is available.")

# Update the Ybus matrix
params["Network"]["Ybus"] = update_Ybus(params)

# Create a Pyomo model
model = pyo.ConcreteModel()

# Define sets
model.Nodes = pyo.RangeSet(1, params["Node_Num"])
model.Gens = pyo.RangeSet(1, params["Unit_Num"])
model.Horizon = pyo.RangeSet(1, params["Horizon"])

# Define variables
model.Pgen = pyo.Var(model.Gens, model.Horizon, within=pyo.NonNegativeReals, initialize=0.0)
model.Qgen = pyo.Var(model.Gens, model.Horizon, within=pyo.Reals, initialize=0.0)
model.WP_p = pyo.Var(model.Horizon, within=pyo.NonNegativeReals)
model.WP_q = pyo.Var(model.Horizon, within=pyo.Reals, initialize=0.0)
model.V = pyo.Var(model.Nodes, model.Horizon, within=pyo.Reals, initialize=1.0)
model.theta = pyo.Var(model.Nodes, model.Horizon, within=pyo.Reals, initialize=0.0)
model.Pflow = pyo.Var(model.Nodes, model.Nodes, model.Horizon, within=pyo.Reals, initialize=0.0)
model.Qflow = pyo.Var(model.Nodes, model.Nodes, model.Horizon, within=pyo.Reals, initialize=0.0)

# Define the objective function
def objective_function(m):
    return sum(
        params["Gen"][g - 1]["Cost_alpha"] * m.Pgen[g, t] ** 2
        + params["Gen"][g - 1]["Cost_beta"] * m.Pgen[g, t]
        + params["Gen"][g - 1]["Cost_gamma"]
        for g in m.Gens
        for t in m.Horizon
    )

model.Obj = pyo.Objective(rule=objective_function, sense=pyo.minimize)

# Define constraints
def PmaxLimit_rule(m, i, t):
    return m.Pgen[i, t] <= params["Gen"][i - 1]["Pmax"]

model.PmaxLimit = pyo.Constraint(model.Gens, model.Horizon, rule=PmaxLimit_rule)

def PminLimit_rule(m, i, t):
    return params["Gen"][i - 1]["Pmin"] <= m.Pgen[i, t]

model.PminLimit = pyo.Constraint(model.Gens, model.Horizon, rule=PminLimit_rule)

def QmaxLimit_rule(m, i, t):
    return m.Qgen[i, t] <= params["Gen"][i - 1]["Qmax"]

model.QmaxLimit = pyo.Constraint(model.Gens, model.Horizon, rule=QmaxLimit_rule)

def QminLimit_rule(m, i, t):
    return params["Gen"][i - 1]["Qmin"] <= m.Qgen[i, t]

model.QminLimit = pyo.Constraint(model.Gens, model.Horizon, rule=QminLimit_rule)

def WPmaxLimit_rule(m, t):
    return (m.WP_p[t] ** 2 + m.WP_q[t] ** 2) <= params["Wind_P"][t - 1] ** 2

model.WPmaxLimit_rule = pyo.Constraint(model.Horizon, rule=WPmaxLimit_rule)

def node_balance_P_rule(m, n, t):
    return (
        params["Network"]["Node"][n - 1]["LoadFactors"] * params["SystemLoad_P"]
        - sum(m.Pgen[g, t] if g == n else 0 for g in model.Gens)
        + sum(m.Pflow[n, l, t] for l in m.Nodes if l != n)
        - m.WP_p[t] * params["WP2Node"][n - 1]
    ) == 0

model.NodeBalanceP = pyo.Constraint(model.Nodes, model.Horizon, rule=node_balance_P_rule)

def node_balance_Q_rule(m, n, t):
    return (
        params["Network"]["Node"][n - 1]["LoadFactors"] * params["SystemLoad_Q"]
        - sum(m.Qgen[g, t] if g == n else 0 for g in model.Gens)
        + sum(m.Qflow[n, l, t] for l in m.Nodes if l != n)
        - m.WP_q[t] * params["WP2Node"][n - 1]
    ) == 0

model.NodeBalanceQ = pyo.Constraint(model.Nodes, model.Horizon, rule=node_balance_Q_rule)

def power_flow_P_rule(m, i, j, t):
    if i != j:
        Y = params["Network"]["Ybus"][i - 1][j - 1]
        return m.Pflow[i, j, t] == (
            m.V[i, t] * m.V[j, t] * params["Sbase"]
            * (-Y.real * pyo.cos(m.theta[i, t] - m.theta[j, t]) - Y.imag * pyo.sin(m.theta[i, t] - m.theta[j, t]))
            + m.V[i, t] * m.V[i, t] * Y.real * params["Sbase"]
        )
    else:
        return pyo.Constraint.Skip

model.PowerFlowP = pyo.Constraint(model.Nodes, model.Nodes, model.Horizon, rule=lambda m, i, j, t: power_flow_P_rule(m, i, j, t))

def power_flow_Q_rule(m, i, j, t):
    if i != j:
        Y = params["Network"]["Ybus"][i - 1][j - 1]
        return m.Qflow[i, j, t] == (
            m.V[i, t] * m.V[j, t] * params["Sbase"]
            * (-Y.real * pyo.sin(m.theta[i, t] - m.theta[j, t]) + Y.imag * pyo.cos(m.theta[i, t] - m.theta[j, t]))
            - m.V[i, t] * m.V[i, t] * Y.imag * params["Sbase"]
        )
    else:
        return pyo.Constraint.Skip

model.PowerFlowQ = pyo.Constraint(model.Nodes, model.Nodes, model.Horizon, rule=lambda m, i, j, t: power_flow_Q_rule(m, i, j, t))

# Transmission capacity limits
def line_capacity_rule(m, i, j, t):
    if i != j:
        return (m.Pflow[i, j, t] ** 2 + m.Qflow[i, j, t] ** 2) <= params["Network"]["Line"][i - 1]["LineCapacity"] ** 2
    else:
        return pyo.Constraint.Skip

model.LineCapacity = pyo.Constraint(model.Nodes, model.Nodes, model.Horizon, rule=line_capacity_rule)

# Wind Power Generation Constraints
def wind_power_generation_rule(m, t):
    return m.WP_q[t] == np.tan(np.arccos(params["power_factor"])) * m.WP_p[t]

model.WindPower = pyo.Constraint(model.Horizon, rule=wind_power_generation_rule)

# Voltage limits
def voltage_lower_limit_rule(m, n, t):
    return params["Network"]["Node"][n - 1]["Vmin"] <= m.V[n, t]

model.VoltageLowerLimit = pyo.Constraint(model.Nodes, model.Horizon, rule=voltage_lower_limit_rule)

def voltage_upper_limit_rule(m, n, t):
    return m.V[n, t] <= params["Network"]["Node"][n - 1]["Vmax"]

model.VoltageUpperLimit = pyo.Constraint(model.Nodes, model.Horizon, rule=voltage_upper_limit_rule)

# Angle limits
def angle_lower_limit_rule(m, n, t):
    return -np.pi / 2 <= m.theta[n, t]

model.AngleLowerLimit = pyo.Constraint(model.Nodes, model.Horizon, rule=angle_lower_limit_rule)

def angle_upper_limit_rule(m, n, t):
    return m.theta[n, t] <= np.pi / 2

model.AngleUpperLimit = pyo.Constraint(model.Nodes, model.Horizon, rule=angle_upper_limit_rule)

def AngleSlackBus_rule(m, t):
    return m.theta[params["refnode"], t] == 0

model.AngleSlackBus = pyo.Constraint(model.Horizon, rule=AngleSlackBus_rule)

def VoltageSlackBus_rule(m, t):
    return m.V[params["refnode"], t] == 1

model.VoltageSlackBus = pyo.Constraint(model.Horizon, rule=VoltageSlackBus_rule)

def check_violations(model, tolerance=1e-6):
    for c in model.component_objects(pyo.Constraint, active=True):
        cobject = getattr(model, str(c))
        for index in cobject:
            body_value = pyo.value(cobject[index].body)
            lower_value = pyo.value(cobject[index].lower) if cobject[index].lower is not None else None
            upper_value = pyo.value(cobject[index].upper) if cobject[index].upper is not None else None

            if lower_value is not None and body_value < lower_value - tolerance:
                print(f"Constraint {c} at index {index} is violated: {body_value} < {lower_value}")
            if upper_value is not None and body_value > upper_value + tolerance:
                print(f"Constraint {c} at index {index} is violated: {body_value} > {upper_value}")

def extract_lagrange_values(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()
    
    iteration_values = []
    lagrange_pattern = re.compile(r'Lagrangian\s+at\s+iteration\s+(\d+):\s+([eE0-9.-]+)')
    
    for line in lines:
        match = lagrange_pattern.search(line)
        if match:
            iteration = int(match.group(1))
            lagrange_value = float(match.group(2))
            iteration_values.append((iteration, lagrange_value))
    
    return iteration_values

# Solve the model
solver = pyo.SolverFactory("ipopt")
solver.options["tol"] = 1e-6
solver.options["max_iter"] = 10000
solver.options["print_level"] = 8

results = solver.solve(model, tee=True, keepfiles=True, logfile="results.csv")

if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
    print("Total system cost: =", pyo.value(model.Obj))
    
    # Extract and display values    
    for t in model.Horizon:
        for i in model.Gens:
            print(f"Unit {i} Power (Real, Reactive) at Hour {t}: {pyo.value(model.Pgen[i, t])} MW, {pyo.value(model.Qgen[i, t])} MVAr")
        print(f"Wind Farm Power (Real, Reactive) at Hour {t}: {pyo.value(model.WP_p[t])} MW, {pyo.value(model.WP_q[t])} MVAr")
        for i in model.Nodes:
            for j in model.Nodes:
                if i != j:
                    print(f"Line {i}-{j} Power Flow (Real, Reactive) at Hour {t}: {pyo.value(model.Pflow[i, j, t])} MW, {pyo.value(model.Qflow[i, j, t])} MVAr")
        for i in model.Nodes:
            print(f"Node {i} Voltage and Angle at Hour {t}: {pyo.value(model.V[i, t])} p.u., {pyo.value(model.theta[i, t])} radians")

    # Define the path to your input CSV file
    input_csv_file_path = "results.csv"
    # Define the path to your output CSV file
    output_csv_file_path = "iterations.csv"

    # Read the CSV file into a list of strings
    with open(input_csv_file_path, "r") as file:
        lines = file.readlines()

    pgen_values = []
    objective_values = []

    # Initialize a flag to track when the next line should be appended
    append_next_line = False

    for i, line in enumerate(lines):
        if solver.options["print_level"] == 8:
            if line.startswith("curr_x[   21]="):
                pgen_values.append(line)
            if line.startswith("curr_x[   22]="):
                pgen_values.append(line)
            if append_next_line:
                objective_values.append(line)
                append_next_line = False
            if line.startswith("iter"):
                append_next_line = True
        if solver.options["print_level"] == 5:
            if append_next_line:
                objective_values.append(line)
                if lines[i + 1].startswith("iter") or lines[i + 1].strip() == "":
                    append_next_line = False
            if line.startswith("iter"):
                append_next_line = True

    with open("pgen_values.csv", "w") as file:
        for line in pgen_values:
            file.write(line)

    with open("iterations.csv", "w") as file:
        for line in objective_values:
            file.write(line)

else:
    print("The optimization problem did not converge to an optimal solution.")
    print("Solver Status:", results.solver.status)
    print("Termination Condition:", results.solver.termination_condition)
    check_violations(model)
