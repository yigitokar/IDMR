# Purpose: Run PB-CLIrunner.py for different hyper params and convert to latex table
import sys
import os
import re
import pandas as pd

def parse_output(file_path):
    # Initialize variables to store the extracted values
    error_square_values = []
    average_time_values = []

    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line contains the target string and extract the value
            if 'average (over parameters) Error Square :' in line:
                try:
                    value = float(line.split(':')[-1].strip())
                    error_square_values.append(value)
                except ValueError:
                    print(f"Error parsing value from line: {line}")

            elif 'average time' in line:
                try:
                    value = float(line.split()[-1].strip())
                    average_time_values.append(value)
                except ValueError:
                    print(f"Error parsing value from line: {line}")

    return error_square_values, average_time_values

def tuples_to_latex(tuples, headers):
    # headers = ['n', 'd', 'time', 'MSE'] pick from these
    # Convert the list of tuples into a DataFrame
    df = pd.DataFrame(tuples, columns=headers)

    # Convert the DataFrame to a LaTeX table
    latex_table = df.to_latex(index=False, header=True)

    return latex_table

# Initialize
result_list = []
headers = ['MSE', 'time']


# set hyper params
params = dict(seed=5,
              n=100, 
              d=3,
              p=5,
              model="MNL",
              theta_seed=1233,
              num_epochs=5,
              init_mu=None,
              B=5,
              estimator="PBinit")


# call PB-CLI runner 
os.system('python PB-CLIrunner.py '    +
         str(params['seed'])         + ' ' + \
         str(params['n'])            + ' ' + \
         str(params['d'])            + ' ' + \
         str(params['p'])            + ' ' + \
         str(params['model'])        + ' ' + \
         str(params['theta_seed'])   + ' ' + \
         str(params['num_epochs'])   + ' ' + \
         str(params['init_mu'])      + ' ' + \
         str(params['B'])            + ' ' + \
         str(params['estimator'])    + ' ' + \
             ' > raw_output.txt'  )

# append to result list
result_list.append(parse_output('raw_output.txt'))

# set hyper params
# call CLI runner 
# append to result list




# Convert to Latex table 
latex_table = tuples_to_latex(result_list, headers)
