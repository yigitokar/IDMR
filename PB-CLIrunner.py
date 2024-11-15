import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import torch
from tqdm import tqdm
import statsmodels.api as sm
import plotly.graph_objects as go
import cvxpy as cvx
import scipy
import os
import multiprocessing as mp
import time
import concurrent.futures
import functools
import sys
import warnings
from classesv2 import *
from PB import *

if __name__ == "__main__":
    ######################################
    seed = sys.argv[1]
    n = sys.argv[2]

    d = sys.argv[3]
    p = sys.argv[4]

    model = sys.argv[5]
    seed_theta = sys.argv[6]

    num_epochs = sys.argv[7]
    initial_mu = sys.argv[8]
    B = sys.argv[9]
    estimator = sys.argv[10]

    seed = int(seed)
    n = int(n)

    d = int(d)
    p = int(p)

    model = model
    seed_theta = int(seed_theta)

    num_epochs = int(num_epochs)
    initial_mu = initial_mu
    B = int(B)
    estimator = estimator


    ##############################################
    ##############################################
    b_idx_theta_list = [] 
    sq_diff = []
    abs_diff = []
    _diff = []
    time_list = []
    random_start_seed = np.random.randint(10000, size = 1)

    if estimator == 'PBinit':
        for b in range(B):
            print('b : ', b)
            print('model : ', model)
            if model == 'Po':
                print('model is Po')
                data_, muTRUE_vec, thetaTRUE_mat = generate_DATA(seed = random_start_seed+b, n = n, d = d, p = p, model = model, seed_theta = seed_theta)
                normalized_thetaTRUE_mat = normalize(thetaTRUE_mat)
                PB_mdl = MDR_v11(textData_obj= data_)

                try:
                    start = time.time()
                    normalized_theta_hat, theta_hat = PB_mdl.PARALLEL_PairwiseBinomial_fit(num_epochs , verbose = False)
                    end = time.time()
                    time_took = end - start
                    print("time spent in 1 Bootstrap sample, (DGP model='Po') ",time_took)
                except:
                    print("Oops!", sys.exc_info()[0], "occurred.")
                    print("Skipping this seed.")
                    continue
            elif model == 'mixture':

                data_, thetaTRUE_mat = generate_DATA(seed = random_start_seed+b, n = n, d = d, p = p, model = 'mixture', seed_theta = seed_theta)
                normalized_thetaTRUE_mat = normalize(thetaTRUE_mat)
                PB_mdl = MDR_v11(textData_obj= data_)

                try:
                    start = time.time()
                    normalized_theta_hat, theta_hat = PB_mdl.PARALLEL_PairwiseBinomial_fit(num_epochs , verbose = False)
                    end = time.time()
                    time_took = end - start
                    print("time spent in 1 Bootstrap sample, (DGP model='mixture') ",time_took)
                except:
                    print("Oops!", sys.exc_info()[0], "occurred.")
                    print("Skipping this seed.")
                    continue
            else:
                print('in PB-CLIrunner.py, model is MNL')
                print('seed  : ', random_start_seed+b)
                data_, thetaTRUE_mat = generate_DATA(seed =random_start_seed+b, n = n, d = d, p = p, model = model, seed_theta= seed_theta)
                normalized_thetaTRUE_mat = normalize(thetaTRUE_mat)
                PB_mdl = MDR_v11(textData_obj= data_)
                try:
                    start = time.time()
                    normalized_theta_hat, theta_hat = PB_mdl.PARALLEL_PairwiseBinomial_fit(num_epochs , verbose = False)
                    end = time.time()
                    time_took = end - start
                    print("time spent in 1 Bootstrap sample, (DGP model='MNL') : ",time_took)
                except:
                    print("Oops!", sys.exc_info()[0], "occurred.")
                    print("Skipping this seed.")
                    continue
        
            b_idx_theta_list.append(normalized_theta_hat)
            sq_diff.append( (normalized_theta_hat - normalized_thetaTRUE_mat)**2 )
            abs_diff.append(  np.average(np.abs(normalized_theta_hat - normalized_thetaTRUE_mat) ))
            _diff.append(  np.average(normalized_theta_hat - normalized_thetaTRUE_mat))
            time_list.append(time_took)
                
    ###### END AND PRINTING
    print('hyperparameters : \n ', 
    '\nseed : ', random_start_seed, 
    '\nn : ', n,
    '\nd : ', d,
    '\np : ', p,
    '\nmodel : ' , model,
    '\nseed_theta : ', seed_theta,
    '\nnum_epochs : ', num_epochs,
    '\ninitial_mu : ', initial_mu,
    '\nB : ', B,
    '\nEstimator : ', estimator)

    print('number of estimations B : ', B)
    print("MSE : "  + str(np.asarray(sq_diff,  dtype=np.float32).mean()))

    sq_diff_11 = np.mean([matrix[1, 1] for matrix in sq_diff])
    print("MSE of theta_11 : "  , sq_diff_11)
    
    print(" bias : "  + str(np.asarray(_diff,  dtype=np.float32).mean()))
    print(" Variance : "  + str(get_variance_from_b_idx_theta_list(b_idx_theta_list)))
    print("average  absolute difference : " + str(np.asarray(abs_diff, dtype=np.float32).mean()))
    print("average time : " + str(np.mean(time_list)))