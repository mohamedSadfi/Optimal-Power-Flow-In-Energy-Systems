# Optimal Power Flow in Energy Problems

## Description

This project is about solving the optimal power flow optimization problem using the Pyomo framework. It involves modeling a simplified electric grid, defining the optimization problem, and visualizing the results.

## Repository Structure

- **main.py**: This is where the optimization problem is declared and solved.
- **system.py**: Contains functions to define larger systems than the one in `params.py`.
- **params.py**: Defines the system with a dictionary containing all grid parameters.

```plaintext
.
├── src
│   ├── main.py
│   ├── system.py
│   ├── params.py
├── README.md

```

## Installation and Running Guide

### Step 1: Clone the Repository

```sh

git clone https://github.com/your-username/OPF_OMFE.git
cd OPF_OMFE

```

### Step 2: Set Up a Virtual Environment

It is advised to create a virtual environment using conda

```sh

conda create --name opf_env python=3.9
conda activate opf_env

```

### Step 3: Install Required Packages

Install Pyomo and other dependencies:

```sh

pip install pyomo numpy

```

### Step 4: Install the Ipopt Solver

Download and install the Ipopt solver:

1. Go to the [Ipopt website](https://coin-or.github.io/Ipopt/).
2. Follow the instructions to download and install the solver for your operating system.
3. Make sure the `ipopt` executable is in your system's PATH.

For example, on Windows:

- Download the precompiled binary from the Ipopt website.
- Extract the downloaded file to a directory of your choice.
- Add the directory containing `ipopt.exe` to your system's PATH:
  - Open the Start Search, type in "env", and select "Edit the system environment variables".
  - In the System Properties window, click on the "Environment Variables" button.
  - In the Environment Variables window, select the PATH variable in the "System variables" section and click "Edit".
  - Click "New" and add the path to the directory where `ipopt.exe` is located.
  - Click "OK" to close all the windows.

For example, on macOS/Linux:

```sh
wget https://www.coin-or.org/download/binary/Ipopt/Ipopt-3.13.4-MacOS.tgz
tar -xzvf Ipopt-3.13.4-MacOS.tgz
sudo mv Ipopt-3.13.4 /opt/ipopt
export PATH=$PATH:/opt/ipopt/bin

```

### Step 5: Run the `main.py` File

Navigate to the `src` directory and run the `main.py` file to solve the optimization problem:

```sh
cd src
python main.py

```

### Contributors

- Mohamed Sadfi
- Jannick Matter
