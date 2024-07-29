""" Function to extract a variable from the results file of the optimization and save it to a csv file """

import csv
import numpy as np


def extract_variable_from_file(input_file_name, output_file_name, index):
    with open(input_file_name, "r") as file:
        lines = file.readlines()
        
    variables = []
    
    for line in lines:
            if line.startswith(f"curr_x[{index:5d}]="):
                variables.append(line)
    
    with open(output_file_name, "w") as file:
        for line in variables:
            file.write(line)
            

""" Function to read the variable value from the csv file and save it in an array """

def save_variable_to_array(input_file_name):
    
    variable = []
    
    with open(input_file_name, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            value = float(row[0].split("=")[1])
            variable.append(value)
            
    return np.array(variable)