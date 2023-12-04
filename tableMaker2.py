# Purpose: Run PB-CLIrunner.py for different hyper params and convert to latex table
import sys
import os
import re
import pandas as pd

def parse_output(file_path):
    # Initialize variables to store the extracted values
    error_square_values = []
    average_time_values = []
    bias_values = []


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

            elif 'bias' in line:
                try:
                    value = float(line.split()[-1].strip())
                    bias_values.append(value)
                except ValueError:
                    print(f"Error parsing value from line: {line}")

    return error_square_values, average_time_values, bias_values

def PB_CLIrunner_wrapper_table2(n,d,estimator):
    params['n'] = n
    params['d'] = d

    if estimator == 'PBinit':
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
    
    else:
        os.system('python CLIrunner.py '    +
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
        
def create_latex_table(data, n_list, d_list):
    # Initialize a dictionary to store data
    data_dict = {d: {} for d in d_list}

    # Populate the dictionary with your data
    for (n, d), (mse, runtime, bias) in data.items():
        data_dict[d][n] = f"({mse}, {bias}, {runtime})"

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=n_list)

    # Convert the DataFrame to a LaTeX table
    latex_table = df.to_latex(escape=False)

    return latex_table

def create_data_dict(n_list, d_list, result_list):
    data_dict = {}
    result_index = 0

    for n in n_list:
        for d in d_list:
            # Ensure there is no index out of range error
            if result_index < len(result_list):
                data_dict[(n, d)] = result_list[result_index]
                result_index += 1
            else:
                break

    return data_dict



# Initialize
params = dict(seed=5,
              n=None, 
              d=None,
              p=5,
              model="MNL",
              theta_seed=1233,
              num_epochs=5,
              init_mu="logm",
              B=5,
              estimator="PBinit")

result_list = []
headers = ['MSE', 'bias', 'time']

n_list = [500, 1000, 1500, 2000, 4000, 6000, 8000, 10000]
d_list = [10, 20, 50]

estimator = 'PBinit'

#Compute table entries
for n in n_list:
    print('table column :', n)
    for d in d_list:
        print('table row :', d)
        PB_CLIrunner_wrapper_table2(n,d,estimator) # set hyper params, calls PB-CLIrunner.py, prints to raw_output.txt
        result_list.append(parse_output('raw_output.txt')) # parse it from raw_output append to result list


# Convert to Latex table 

data_dict = create_data_dict(n_list, d_list, result_list)

latex_table = create_latex_table(data_dict, n_list, d_list)

print(latex_table)
