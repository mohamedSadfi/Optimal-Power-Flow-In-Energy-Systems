# Optimal Power Flow 


## Description
I tried with simulated annealing, but it is almost impossible to do for the optimal power flow problem.
SO I first tried to solve the problem using the pyomo optimization tool using the interior point method. When that works I want to try to implement my own interior point solver to compare it with the pyomo solver. 

## Make the Shit Run
First create a conda environment and install pyomo
You must install ipopt as well. Run this conda install -c conda-forge ipopt

# Files 
interior_point.py

# Todos 

- Make a detailed data analysis:
    - compute the lagrange function at each iteration and plot it 
    - Explain complexity of the interior point method and explain scalibility to realistic system (Do example for the zurich electric grid)
- Explain the interior point method on the report
- Make a detailed motivation for the AC optimal power flow optimization and why it is important to solve
- Delete the implementation part of the report
- Investigate the performance of the algorithm at the end: Can we parallelize the thing, etc. ? 
- Implement my own interior point solver !!!!!!!!!!!!!!!
- Make the interior_point pyomo file easier to handle ( change the nu,ber of units, of loads, etc.)

