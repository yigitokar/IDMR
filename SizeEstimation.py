import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import torch
from tqdm import tqdm
import statsmodels.api as sm
import plotly.graph_objects as go
import cvxpy as cvx
import scipy
import multiprocessing as mp
import time
import concurrent.futures
import functools
import pickle
import sys
from classesv2 import *
from PB import *

from scipy.stats import t

import os
os.environ["MOSEK_LICENSE_FILE"] = "/teamspace/studios/this_studio/mosek.lic"


def parametric_bootstrap_SE_for_index(theta_hat, B, element_index, n, d, p, num_epochs, model, seed):
    bootstrap_theta_hats = []
    b = 0
    while len(bootstrap_theta_hats) < B:
        print('              b = ', b, 'out of ', B)
        try:
            data_hat = generate_DATA_4BS(theta_hat, seed=seed+b, n=n, d=d, p=p, model=model)
            mdl_hat_hat = MDR_v11(textData_obj=data_hat)
            normalized_theta_hat_hat, theta_hat_hat = mdl_hat_hat.PARALLEL_PairwiseBinomial_fit(num_epochs, verbose=False)
            bootstrap_theta_hats.append(normalized_theta_hat_hat)
        except:
            print("Oops!", sys.exc_info()[0], "occurred.")
            print("Skipping this seed in the bootstrap function.")
        b += 1

    # Extract the element of the theta_hat matrix that is of interest
    values = [theta_hat[element_index[0], element_index[1]] for theta_hat in bootstrap_theta_hats]

    # Compute standard deviation of parametric bootstrap estimates
    std_err = np.std(values)
    print('Standard deviation of Parametric Bootstrap theta_hats, for index', element_index, ' : ', std_err, 'B = ', B)
    return std_err
def get_t_test(normalized_theta_hat_index, std_err, alpha, n):
    # Compute the t-statistic 
    t_stat = (normalized_theta_hat_index - 0) / std_err
    p_value = 2 * (1 - t.cdf(abs(t_stat), df= n - 1))

    # Check if we reject H0
    if p_value < alpha:
        print("        Reject null hypothesis, theta_hat[1,1] which is ",np.round(normalized_theta_hat_index, 5), "is significantly different from 0 (WRONG)")

        return("reject")
    else:
        print("        (Normalized) theta_hat[1,1] which is ",  np.round(normalized_theta_hat_index, 5),'is close to 0')
        return("accept")

if __name__ == '__main__':

    seed = 1235
    n = 250
    d = 10   
    p = 5
    model = 'MNL'
    seed_theta= 1233
    alpha = 0.05    # Significance level
    B = 100
    num_simulations = 1000 # Number of simulations for size estimation
    num_epochs = 10
    element_index = (1,1)
    deviation = 0.00


    np.random.seed(seed_theta)
    thetaTRUE_mat= np.random.normal(0,1,(p,d))
    normalized_thetaTRUE_mat = normalize(thetaTRUE_mat)
    print(pd.DataFrame(normalized_thetaTRUE_mat))

    #### Burda normalize ettikten sonra theta_H0' icin bir modifikastyon yapiyoruz.
    theta_H0= normalized_thetaTRUE_mat.copy() 
    theta_H0[1,1]= deviation
    print(pd.DataFrame(theta_H0))

    rejections = 0
    acceptances = 0
    for _ in range(num_simulations):
        print('Simulation number ', _)

        # Generate data with theta_H0
        data_H0 = generate_DATA_4BS(theta_H0, seed=seed+_ , n=n, d=d, p=p, model=model) 
        try: 
            mdl = MDR_v11(textData_obj= data_H0)
            normalized_theta_hat, theta_hat = mdl.PARALLEL_PairwiseBinomial_fit(num_epochs , verbose = False)

            print('theta_hat_1_1', normalized_theta_hat[1,1])
            # Fit Parametric Bootstrap to get standard errors for an index of interest   
            std_err = parametric_bootstrap_SE_for_index(B = B, element_index=element_index, theta_hat = theta_hat, n = n, d = d, p = p, num_epochs = num_epochs, model = model, seed = seed)
            # Conduct t-test
            test_result = get_t_test(normalized_theta_hat_index= normalized_theta_hat[1,1], std_err=std_err, alpha = alpha, n = n)
        except:
            print("Oops!", sys.exc_info()[0], "occurred.")
            print("Skipping this seed.")
            continue
    

        if test_result == 'reject':
            rejections += 1
        elif test_result == 'accept':
            acceptances += 1
        print("ESTIMATED SIZE in the for loop : ", rejections/(rejections + acceptances))
    # Estimate size
    estimated_size = rejections / num_simulations
    print(f"Estimated size of the test: {estimated_size:.4f}")


