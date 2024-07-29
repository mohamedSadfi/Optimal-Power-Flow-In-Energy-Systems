
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

import pandas as pd # type: ignore
from system import save_variable_to_array
from plots import plot_variable




with open("iterations.csv", "r") as file:
    lines = file.readlines()

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

df = pd.DataFrame(data_lines, columns=header)

df["objective"] = df["objective"].astype(float)
objective_values = df["objective"].to_numpy()

errors = df["inf_pr"].astype(float).to_numpy()
         
        
pgen1 = save_variable_to_array("pgen1_values.csv")
pgen2 = save_variable_to_array("pgen2_values.csv")

#plot_errors(errors)
#plot_pgen(pgen1, pgen2)
#plot_isocost_lines(pgen1, pgen2)



plt.show()