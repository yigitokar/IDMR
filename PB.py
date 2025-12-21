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
import sys
from classesv2 import *



def get_subdata(data_, k):
    
    C_k = data_.C[:,k] 
    C_d = data_.C[:,0]
    C_sub = np.column_stack((C_d, C_k))

    V_sub = data_.V
    M_sub = C_sub.sum(axis = 1)
    subdata_ = textData(C_sub,V_sub,M_sub)
    return(subdata_)

def nll_l_cvm_cvx(C, V, m, verbose=False, solver='SCS', lambda_=0.0):
    """Multinomial logistic regression via CVXPY with optional L1 penalty.

    Args:
        C: Count matrix (n, d)
        V: Covariate matrix (n, p)
        m: Total counts per observation (n,)
        verbose: Print solver output
        solver: CVXPY solver
        lambda_: L1 regularization strength (default 0.0 = no regularization)

    Returns:
        theta: Estimated parameters, shape (p, d)
    """
    n = C.shape[0]
    d = C.shape[1]
    p = V.shape[1]

    theta_layer = cvx.Variable((p, d), nonneg=False)  # p, d
    eta_mat = V @ theta_layer  # (n, d)

    # Sum over the rows
    eta_vec = cvx.sum(eta_mat, axis=1)

    # Log-sum-exp over the rows
    lse_eta_vec = cvx.log_sum_exp(eta_mat, axis=1)

    q1 = m @ lse_eta_vec
    q2 = cvx.sum(cvx.diag(C @ eta_mat.T))

    nll_loss = q1 - q2

    # Add L1 penalty if lambda_ > 0
    if lambda_ > 0:
        l1_penalty = lambda_ * cvx.norm1(theta_layer)
        objective = nll_loss + l1_penalty
    else:
        objective = nll_loss

    problem = cvx.Problem(cvx.Minimize(objective))
    scs_opts = {
        'max_iters': 2500,
        'eps': 1e-3,
        'alpha': 1.5,
        'scale': 1.0,
        'normalize': True,
        'rho_x': 1e-3
    }
    problem.solve(verbose=verbose, solver=cvx.SCS, **scs_opts)
    return theta_layer.value


def get_theta_k_pairwiseBinomial(k, data_, lambda_=0.0):
    """Pairwise binomial initialization for choice k with optional L1 penalty.

    Args:
        k: Choice index
        data_: textData object
        lambda_: L1 regularization strength (default 0.0 = no regularization)

    Returns:
        theta_k: Estimated parameters for choice k, shape (p,)
    """
    subdata_ = get_subdata(data_, k)

    theta_hat_mle_cvx_SUB = nll_l_cvm_cvx(
        subdata_.C, subdata_.V, subdata_.m,
        verbose=False, solver='MOSEK', lambda_=lambda_
    )
    theta_hat_mle_cvx_SUB[0, 1] = -1 * theta_hat_mle_cvx_SUB[0, 0]
    theta_hat_mle_cvx_SUB[0, 0] = 0

    return theta_hat_mle_cvx_SUB[:, 1]


# partial_pairwiseBinomial_func_cvx=functools.partial(get_theta_k_pairwiseBinomial_cvx,
#                                                 data_ = data_) 


class MDR_v11:
    def __init__(self, textData_obj, lambda_=0.0):
        """Initialize MDR_v11 estimator.

        Args:
            textData_obj: textData object containing C, V, m
            lambda_: L1 regularization strength (default 0.0 = no regularization)
        """
        # get data from textData object
        self.data = textData_obj
        self.C = textData_obj.C
        self.V = textData_obj.V
        self.m = textData_obj.m
        # put the dimensions as attributes so that I dont have to
        # define them every time.
        self.n = self.V.shape[0]
        self.p = self.V.shape[1]
        self.d = self.C.shape[1]

        self.k_grid = list(np.arange(1, self.d))

        self.mu_vec = np.zeros((self.n, 1))

        # L1 regularization strength
        self.lambda_ = lambda_
        
    def initialize(self):
        if self.initial_mu == 'zero':
            self.mu_vec = np.zeros((self.n, 1))
            self.mu_vec_cvx = cvx.Parameter((self.n, 1), nonneg=False)
            self.mu_vec_cvx.value = self.mu_vec

            self.theta_mat = np.zeros((self.p, self.d))
            # compute the likelihood with mu = zeros and theta_hat_mat = poisson cvx
            self.theta_mat = fit_nlcv_with_cvx_v8(
                self.C, self.V, mu_vec=self.mu_vec, lambda_=self.lambda_
            )

            # We expect a lower nll when we do iteration...

        elif self.initial_mu == 'logm':

            self.mu_vec = np.log(self.m)

            self.mu_vec_cvx = cvx.Parameter((self.n, 1), nonneg=False)
            self.mu_vec_cvx.value = self.mu_vec.reshape(-1, 1)
            self.theta_mat = np.zeros((self.p, self.d))

            # compute the likelihood with mu = zeros and theta_hat_mat = poisson cvx
            self.theta_mat = fit_nlcv_with_cvx_v8(
                self.C, self.V, mu_vec=self.mu_vec, lambda_=self.lambda_
            )

        else:
            return('Error: No such start as ' + str(self.initial_mu)) 
        
    def PARALLEL_initialize(self):
        if self.initial_mu == 'zero': 
            self.mu_vec = np.zeros((self.n,1))
            
            self.PARALLEL_update_theta_bar_mat()
            
        elif self.initial_mu == 'logm':
            
            self.mu_vec = np.log(self.m).reshape(-1,1)
            self.PARALLEL_update_theta_bar_mat()
            
        else:
            return('Error: No such start as ' + str(self.initial_mu)) 

    def PARALLEL_initialize_theta_PairwiseBinomial(self):

        print('Initializing the theta by parallel PB')
        start = time.time()
        partial_pairwiseBinomial_func = functools.partial(
            get_theta_k_pairwiseBinomial,
            data_=self.data,
            lambda_=self.lambda_
        )

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(partial_pairwiseBinomial_func, self.k_grid))
        results_array = np.array(results).T
        result_matrix = np.hstack((np.zeros((self.p, 1)), results_array))
        self.theta_mat = result_matrix
        end = time.time()
        time_took = end - start
        print("time spent in 1 PB initialing : ", time_took)


    def PARALLEL_update_mu_bar_vec(self):
        eta_mat = self.V @ self.theta_mat
        eta_mat_tensor = torch.tensor(eta_mat)
        logsumexp_vec = torch.logsumexp(eta_mat_tensor, 1).detach().numpy()
        mu_star_vec = np.log(self.m) - logsumexp_vec

        self.mu_vec = mu_star_vec.reshape(-1,1)

    def GPU_PARALLEL_update_theta_bar_mat(self):
        start = time.perf_counter()
        words = list(np.arange(self.d))

        # Move data to GPU
        C_gpu = torch.tensor(self.C, device='cuda')
        V_gpu = torch.tensor(self.V, device='cuda')
        mu_vec_gpu = torch.tensor(self.mu_vec, device='cuda')

        # Define the CELL_minQ_kn function using PyTorch
        def CELL_minQ_kn_gpu(k):
            Ck_gpu = C_gpu[:, k].reshape(-1, 1)
            theta_k_gpu = torch.zeros((self.p, 1), device='cuda', requires_grad=True)
            optimizer = torch.optim.LBFGS([theta_k_gpu], lr=0.1, max_iter=100)

            def closure():
                optimizer.zero_grad()
                eta_gpu = V_gpu @ theta_k_gpu + mu_vec_gpu
                loss = torch.sum(torch.exp(eta_gpu)) - torch.sum(Ck_gpu * eta_gpu)
                loss.backward()
                return loss

            optimizer.step(closure)
            return theta_k_gpu.detach().cpu().numpy().flatten()

        # Use PyTorch's multiprocessing to parallelize the computation
        with torch.multiprocessing.Pool() as pool:
            theta_list = pool.map(CELL_minQ_kn_gpu, words)

        finish = time.perf_counter()
        print(f'One parallel iteration, Finished in {round(finish-start, 2)} second(s)')

        self.theta_mat = np.reshape(np.ravel(theta_list), (self.p, self.d), order='F')   
        
    def PARALLEL_update_theta_bar_mat(self):

        start = time.perf_counter()
        words = list(np.arange(self.d))
        # Because I am using map method of executor object,
        # the context manager below is equivalent to parallel
        # minimization of Q_kn
        with concurrent.futures.ProcessPoolExecutor() as executor:

            Cell_minQ_k = functools.partial(
                CELL_minQ_kn,
                C=self.C,
                V=self.V,
                mu_vec=self.mu_vec,
                verbose=False,
                solver='SCS',
                lambda_=self.lambda_
            )

            result_list = executor.map(Cell_minQ_k, words)

        finish = time.perf_counter()

        thtList = []
        for i in result_list:
            thtList.append(i)

        self.theta_mat = np.reshape(np.ravel(thtList), (self.p, self.d), order='F')       
        
    def PARALLEL_oneRun(self):
        ''' takes mu_vec^0 and theta_mat^0 --> updates parameters as 
        attributes of the object, mu_vec^1, theta_mat^1
        first update mu given theta,
        then update theta given mu 
        '''
        self.PARALLEL_update_mu_bar_vec()
        self.PARALLEL_update_theta_bar_mat() 
    
    def GPU_PARALLEL_oneRun(self):
        ''' takes mu_vec^0 and theta_mat^0 --> updates parameters as 
        attributes of the object, mu_vec^1, theta_mat^1
        first update mu given theta,
        then update theta given mu 
        '''
        self.PARALLEL_update_mu_bar_vec()
        self.GPU_PARALLEL_update_theta_bar_mat() 



    def PARALLEL_PairwiseBinomial_fit(self, num_epochs , verbose = False):
        '''
        Fits the iterative estimator paralelly, with the 
        initial theta gotten from parallel Pairwise Binomial
        '''
        
        if (num_epochs == 0):
            self.PARALLEL_initialize_theta_PairwiseBinomial()
            return(self.theta_mat)
        
        else:  
            self.num_epochs = num_epochs
            self.verbose = verbose
            self.PARALLEL_initialize_theta_PairwiseBinomial()

            for epoch in range(self.num_epochs): 
                self.PARALLEL_oneRun()

            self.normalized_theta = normalize(self.theta_mat)
            #print(pd.DataFrame(self.normalized_theta))
            return(self.normalized_theta,self.theta_mat)     

    def GPU_PARALLEL_PairwiseBinomial_fit(self, num_epochs , verbose = False):
        '''
        Fits the iterative estimator paralelly, with the 
        initial theta gotten from parallel Pairwise Binomial
        '''
        
        if (num_epochs == 0):
            self.PARALLEL_initialize_theta_PairwiseBinomial()
            return(self.theta_mat)
        
        else:  
            self.num_epochs = num_epochs
            self.verbose = verbose
            self.PARALLEL_initialize_theta_PairwiseBinomial()

            for epoch in range(self.num_epochs): 
                self.GPU_PARALLEL_oneRun()

            self.normalized_theta = normalize(self.theta_mat)
            #print(pd.DataFrame(self.normalized_theta))
            return(self.normalized_theta,self.theta_mat)     
        
        
    def PARALLEL_fit(self, num_epochs, initial_mu , verbose = False):
        '''
        num_epochs is the number of times we will do the iteration
        num_iter is the the number of GD steps to find min of -ll, in each 
        seperate poisson regression
        
        initial_mu should be one of the following strings, 'zero' or 'logm'
        
        '''
        
        if (num_epochs == 0):
            self.initial_mu = initial_mu
            self.num_epochs = num_epochs
            self.verbose = verbose
            self.PARALLEL_initialize()
            
            self.normalized_theta = normalize(self.theta_mat)
            return(self.normalized_theta,self.theta_mat)
        
        else:  
            self.initial_mu = initial_mu
            self.num_epochs = num_epochs
            self.verbose = verbose
            self.PARALLEL_initialize()
            
            for epoch in range(self.num_epochs): 
                self.PARALLEL_oneRun()

            self.normalized_theta = normalize(self.theta_mat)
            return(self.normalized_theta,self.theta_mat)     

    def update_mu_bar_vec(self):
        eta_mat = self.V @ self.theta_mat
        eta_mat_tensor = torch.tensor(eta_mat)
        logsumexp_vec = torch.logsumexp(eta_mat_tensor, 1).detach().numpy()
        mu_star_vec = np.log(self.m) - logsumexp_vec
        
        self.mu_vec = mu_star_vec
        self.mu_vec_cvx.value = self.mu_vec.reshape(-1,1)

    def update_theta_bar_mat(self):
        """Update theta matrix given mu (takes from the object attribute).

        Dimensions should be as follows:
        self.C = n,d
        self.V = n,p
        self.m = (n,)
        self.mu_vec = (n,)
        """
        self.theta_mat = fit_nlcv_with_cvx_v8(
            self.C, self.V, mu_vec=self.mu_vec, lambda_=self.lambda_
        )
    
    def oneRun(self): 
        ''' takes mu_vec^0 and theta_mat^0 --> updates parameters as 
        attributes of the object, mu_vec^1, theta_mat^1
        first update mu given theta,
        then update theta given mu 
        '''
        self.update_mu_bar_vec()
        self.update_theta_bar_mat()   
        
    def get_fd(self, theta_vec, mu_vec, Ck):
        a = np.exp( self.V @ theta_vec + mu_vec)
        fd = self.V.T @ (a - Ck)
        return(fd)
    
        
    def get_sd(self, theta_vec, mu_vec):    
        a = np.exp( self.V @ theta_vec + mu_vec)
        A = a.reshape(-1,1) @ np.ones((1,self.p))
        sd = self.V.T @ ( A * self.V)
        return(sd)
    
    def theta_bar(self, theta_mat):
        eta_mat = self.V @ theta_mat
        eta_mat_tensor = torch.tensor(eta_mat)
        logsumexp_vec = torch.logsumexp(eta_mat_tensor, 1).detach().numpy()
        mu_star_vec = np.log(self.m) - logsumexp_vec        
        return(mu_star_vec)

        
    def fit_NR_poisson(self):
        theta_s1 = self.theta_mat.copy()
        mu_s1 = self.theta_bar(theta_s1)
        
        theta_s = self.theta_mat.copy()*0
        
        for k in range(self.d):
            th_ks1 = theta_s1[:,k]
            Ck = self.C[:,k]
            fd = self.get_fd(th_ks1, mu_s1, Ck)
            sd = self.get_sd(th_ks1, mu_s1)
            theta_s[:,k] = th_ks1 - np.linalg.inv(sd) @ fd

        self.theta_mat = theta_s.copy()
        return(theta_s)
        
    def fit_NewtonRaphson(self, num_iter):
        self.theta_mat = np.zeros((self.p, self.d))
        self.mu_vec = np.log(self.m)
        #self.mu_vec = np.log(self.m)*0
        for itera in range(num_iter):
            print(itera)
            self.theta_mat = self.fit_NR_poisson()
        
        self.normalized_theta = normalize(self.theta_mat)
        return(self.normalized_theta,self.theta_mat)
    
    def fit_Bohning_poisson(self):
        theta_s1 = self.theta_mat.copy()
        mu_s1 = self.theta_bar(theta_s1)
        
        theta_s = self.theta_mat.copy()*0
        XtXinv = np.linalg.inv(np.transpose(self.V) @ self.V)
        E = np.identity(2)
        for k in range(self.d):
            th_ks1 = theta_s1[:,k]
            Ck = self.C[:,k]
            fd = self.get_fd(th_ks1, mu_s1, Ck)
           
            sd_B_inverse = 2 * np.kron( E + np.ones((2, 2)), XtXinv)  ##
            
            theta_s[:,k] = th_ks1 - sd_B_inverse @ fd

        self.theta_mat = theta_s.copy()
        return(theta_s)
    
    def fit_Bohning92(self, num_iter):
        self.theta_mat = np.zeros((self.p, self.d))
        self.mu_vec = np.log(self.m)
        for itera in range(num_iter):
            self.theta_mat = self.fit_Bohning_poisson()
        
        self.normalized_theta = normalize(self.theta_mat)
        return(self.normalized_theta,self.theta_mat)
    
    def fit_MLE(self, num_iter):
        mdl = nll_l_cvm(self.data)
        optimizer = torch.optim.AdamW(mdl.parameters(), lr=1e-3)
        for steps in range(num_iter):
            loss = mdl() 
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        normalized_theta_hat3 = normalize(mdl.theta_layer.weight.transpose(1,0).detach().numpy())
        return normalized_theta_hat3, mdl.theta_layer.weight.transpose(1,0).detach().numpy()
	        
    def fit(self, num_epochs, initial_mu , verbose = False):
        '''
        num_epochs is the number of times we will do the iteration
        num_iter is the the number of GD steps to find min of -ll, in each 
        seperate poisson regression
        
        initial_mu should be one of the following strings, 'zero' or 'logm'
        
        '''
        
        if (num_epochs == 0):
            self.initial_mu = initial_mu
            self.num_epochs = num_epochs
            self.verbose = verbose
            self.initialize()
            
            self.normalized_theta = normalize(self.theta_mat)
            return(self.normalized_theta,self.theta_mat)
        
        else:  
            self.initial_mu = initial_mu
            self.num_epochs = num_epochs
            self.verbose = verbose
            self.initialize()
            
            for epoch in range(self.num_epochs): 
                self.oneRun()

            self.normalized_theta = normalize(self.theta_mat)
            return(self.normalized_theta,self.theta_mat)