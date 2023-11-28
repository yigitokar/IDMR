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

parse_output('raw_output.txt')