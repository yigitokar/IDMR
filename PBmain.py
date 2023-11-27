import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import torch
from tqdm import tqdm
import statsmodels.api as sm
import plotly.graph_objects as go
import cvxpy as cvx
import scipy
import ot as ot
import multiprocessing as mp
import time
import concurrent.futures
import functools
import sys
from classesv2 import *
from PB import *

if __name__ == "__main__":

	seed = 2
	n = 100
	d = 10 
	p = 5
	model = 'MNL'
	seed_theta= 1233
	num_epochs = 10
	k_grid = list(np.arange(1,d))
	initial_mu = 'zero'	

	data_ ,thetaTRUE_mat = generate_DATA(seed = seed, n = n, d = d, p = p, model = model, seed_theta = seed_theta)
	normalized_thetaTRUE_mat = normalize(thetaTRUE_mat)
	print("Normalized Theta TRUE MAT : \n", pd.DataFrame(normalized_thetaTRUE_mat))



	PB_mdl = MDR_v11(textData_obj= data_)

	normalized_theta_hat, theta_hat = PB_mdl.PARALLEL_PairwiseBinomial_fit(num_epochs , verbose = False)
	print('MSE with vanilla-PB init : ', MSE_thetas(normalized_thetaTRUE_mat, normalized_theta_hat))

	mdl = MDR_v11(textData_obj= data_)

	normalized_theta_hat2, theta_hat2 = mdl.PARALLEL_fit( num_epochs, initial_mu , verbose = False)
	print('MSE with logm init : ', MSE_thetas(normalized_thetaTRUE_mat, normalized_theta_hat2))

	normalized_theta_hat3, theta_hat3 = mdl.fit_MLE( num_iter = 10000)
	print('MSE with MLE : ', MSE_thetas(normalized_thetaTRUE_mat, normalized_theta_hat3))

	#n_grid = [100, 300, 500]
	n_grid = [100,300, 500, 700, 1000,1200, 1400, 1600]

	MSE_PBinit = []
	MSE_PBinit22 = []
	MSE_PBinit33 = []
	MSE_PB = []
	MSE_initZero5 = []
	MSE_initZero10 = []
	MSE_MLE = []

	for n in n_grid:
		print('n : ', n)
		data_ ,thetaTRUE_mat = generate_DATA(seed = seed, n = n, d = d, p = p, model = model, seed_theta = seed_theta)
		normalized_thetaTRUE_mat = normalize(thetaTRUE_mat)

		PB_mdl = MDR_v11(textData_obj= data_)
		normalized_theta_hat, theta_hat = PB_mdl.PARALLEL_PairwiseBinomial_fit(num_epochs = 5 , verbose = False)

		PB_mdl = MDR_v11(textData_obj= data_)
		normalized_theta_hat22, theta_hat = PB_mdl.PARALLEL_PairwiseBinomial_fit(num_epochs = 1 , verbose = False)

		PB_mdl = MDR_v11(textData_obj= data_)
		normalized_theta_hat33, theta_hat = PB_mdl.PARALLEL_PairwiseBinomial_fit(num_epochs = 3, verbose = False)

		PB_mdl = MDR_v11(textData_obj= data_)
		normalized_theta_hatPB = PB_mdl.PARALLEL_PairwiseBinomial_fit( num_epochs = 0, verbose = False)

		mdl = MDR_v11(textData_obj= data_)
		normalized_theta_hat_NC_5, theta_hat2 = mdl.PARALLEL_fit( num_epochs = 5, initial_mu = 'logm' , verbose = False)

		mdl = MDR_v11(textData_obj= data_)
		normalized_theta_hat_NC_10, theta_hat2 = mdl.PARALLEL_fit( num_epochs = 10, initial_mu = 'logm' , verbose = False)

		mdl = MDR_v11(textData_obj= data_)
		normalized_theta_hat_MLE, theta_hat3 = mdl.fit_MLE( num_iter = 10000)

		err_PBinit = MSE_thetas(normalized_thetaTRUE_mat, normalized_theta_hat)
		err_PBinit22 = MSE_thetas(normalized_thetaTRUE_mat, normalized_theta_hat22)
		err_PBinit33 = MSE_thetas(normalized_thetaTRUE_mat, normalized_theta_hat33)

		err_PB = MSE_thetas(normalized_thetaTRUE_mat, normalized_theta_hatPB)
		err_initZero5 = MSE_thetas(normalized_thetaTRUE_mat, normalized_theta_hat_NC_5)
		err_initZero10 = MSE_thetas(normalized_thetaTRUE_mat, normalized_theta_hat_NC_10)
		err_MLE = MSE_thetas(normalized_thetaTRUE_mat, normalized_theta_hat_MLE)

		MSE_PBinit.append(err_PBinit)
		MSE_PBinit22.append(err_PBinit22)
		MSE_PBinit33.append(err_PBinit33)


		MSE_PB.append(err_PB)
		MSE_initZero5.append(err_initZero5)
		MSE_initZero10.append(err_initZero10)
		MSE_MLE.append(err_MLE)


	# Plotting the lines
	plt.plot(n_grid, MSE_PB, label='Pairwise Binomial', color='black')
	plt.plot(n_grid, MSE_PBinit22, label='Init: PB, S = 1', color='brown')
	plt.plot(n_grid, MSE_PBinit33, label='Init: PB, S = 3', color='purple')
	plt.plot(n_grid, MSE_PBinit, label='Init: PB, S = 5', color='orange')
	
	
	plt.plot(n_grid, MSE_MLE, label='Full MLE', color='red')

	# Adding a title and labels
	plt.title("MSE vs n | d = 10")
	plt.xlabel("n")
	plt.ylabel("MSE")

	# Display the legend
	plt.legend()

	# Show the plot
	plt.show()


	# Plotting the lines
	plt.plot(n_grid, MSE_PB, label='Pairwise Binomial', color='black')
	plt.plot(n_grid, MSE_initZero5, label='Init: mu = logm, S = 5', color='brown')
	plt.plot(n_grid, MSE_initZero10, label='Init: mu = logm, S = 10', color='orange')
	plt.plot(n_grid, MSE_MLE, label='Full MLE', color='red')

	# Adding a title and labels
	plt.title("MSE vs n | d = 10")
	plt.xlabel("n")
	plt.ylabel("MSE")

	# Display the legend
	plt.legend()

	# Show the plot
	plt.show()






	#PB_mdl.PARALLEL_initialize_theta_PairwiseBinomial()


	# partial_pairwiseBinomial_func=functools.partial(get_theta_k_pairwiseBinomial,
    #                                             data_ = data_,
    #                                             num_steps_autograd = 10000) 

	# with concurrent.futures.ProcessPoolExecutor() as executor:
	# 	results = list(executor.map(partial_pairwiseBinomial_func, k_grid))

	# results_array = np.array(results).T  
	# result_matrix = np.hstack((np.zeros((p, 1)), results_array))
	# print(pd.DataFrame(result_matrix))

