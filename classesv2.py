import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from tqdm import tqdm
import statsmodels.api as sm
import plotly.graph_objects as go
import cvxpy as cvx
import scipy
import multiprocessing as mp
import time
import concurrent.futures
import functools
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.stats import norm



torch.manual_seed(1337)
class nll_l_cvm(nn.Module):

    def __init__(self, data_):
        super().__init__()
        self.C = torch.from_numpy(data_.C).to(torch.float64) #(n,d)
        self.V = torch.from_numpy(data_.V).to(torch.float64) #(n,p)
        self.m = torch.from_numpy(data_.m).to(torch.float64) #(m,)
        
        self.n = self.C.shape[0]
        self.d = self.C.shape[1]
        self.p = self.V.shape[1]
        
        self.theta_layer = nn.Linear(self.p, self.d, dtype=torch.float64 ,bias=False)

    def forward(self):
        eta_mat = self.theta_layer(self.V) #(n,d)
        eta_vec = torch.sum(eta_mat, 1 )
    
        lse_eta_vec = torch.logsumexp(eta_mat,axis = 1)
    
        q1 = self.m @ lse_eta_vec
        q2 =  torch.sum(torch.diag(self.C@(torch.transpose(eta_mat,1,0))))
        loss = q1 - q2 

        return(loss/self.n)

def get_variance_from_b_idx_theta_list(b_idx_theta_list):
    tensor = np.stack(b_idx_theta_list)
    var_mat = np.var(tensor, axis = 0 )
    average_over_parameters_Variance = np.mean(var_mat)
    return(average_over_parameters_Variance)
# print('Imported classes module')

def MSE_thetas(thetaTRUE_mat, theta_mat):
    result = ((thetaTRUE_mat - theta_mat)**2).mean()
    return(result)

def CELL_minQ_kn(k,C, V, mu_vec, verbose = False, solver = 'SCS'):
    
    n = C.shape[0]
    d = C.shape[1]
    p = V.shape[1]
    
    theta_vec_k = cvx.Variable((p,1), nonneg=False)
    q1 = cvx.exp(V @ theta_vec_k + mu_vec)
    q2 = cvx.multiply(C[:,k].reshape(-1,1) , (V @ theta_vec_k + mu_vec) )
    o_ = cvx.sum(q1 - q2 )

    objective =  o_
        
    problem = cvx.Problem(cvx.Minimize(objective))
    problem.solve(verbose=False, solver=cvx.SCS)
    return(theta_vec_k.value)
    
    

def do_something(seconds):
    time.sleep(1)
    print(seconds)

def fit_nlcv_with_cvx_v8(C, V, mu_vec, verbose = False, solver = 'MOSEK'):
    '''C is a column matrix, V, m are np arrays, mu_vec is a np array in this version.'''
    # Bu fonksiyonu MDRv7'nin icine koy!
    solver = 'MOSEK'
    verbose = False
    n = C.shape[0]
    d = C.shape[1]
    p = V.shape[1]

    mu_vec_mat = mu_vec.reshape((-1,1)) @  np.ones((1,d)) # repeats mu d times

    mu_vec_mat_cvx= cvx.Parameter((n,d),nonneg=False)
    mu_vec_mat_cvx.value = mu_vec_mat

    theta_mat = cvx.Variable((p,d), nonneg=False)
    # theta_mat.value = thetaTRUE_mat ###### computes correct lcv, checked manually.

    q1 = cvx.exp(V @ theta_mat + mu_vec_mat)
    q2 = cvx.multiply(C , (V @ theta_mat + mu_vec_mat) )
    o_ = cvx.sum(q1 - q2 )

    objective =  o_
    
    #constraints = [theta_mat[:,0] == 0]
    
    problem = cvx.Problem(cvx.Minimize(objective))
    problem.solve(verbose=verbose, solver=cvx.MOSEK)
    return(theta_mat.value)


def normalize(theta):
    p = theta.shape[0]
    d = theta.shape[1]

    fixed_param = theta[:,0]
    theta_normalized = np.zeros((p,d))
    for j in np.arange(0,d):
        theta_normalized[:,j] = theta[:,j] - fixed_param
    return(theta_normalized)


# Newest version that has polarizations and Newton-Raphson.
class MDR_v10:
    def __init__(self, textData_obj):
        # get data from textData object
        self.data = textData_obj
        self.C= textData_obj.C
        self.V = textData_obj.V
        self.m = textData_obj.m
        # put the dimensions as attributes so that I dont have to
        # define them every time.
        self.n = self.V.shape[0]
        self.p = self.V.shape[1]
        self.d = self.C.shape[1]
        
        self.mu_vec = np.zeros((self.n,1))
        
    def initialize(self):
        if self.initial_mu == 'zero': 
            self.mu_vec = np.zeros((self.n,1))
            self.mu_vec_cvx = cvx.Parameter((self.n,1),nonneg=False)
            self.mu_vec_cvx.value = self.mu_vec
            
            self.theta_mat = np.zeros((self.p, self.d))
            # compute the likelihood with mu = zeros and theta_hat_mat = poisson cvx
            self.theta_mat = fit_nlcv_with_cvx_v8(self.C,self.V,mu_vec = self.mu_vec)
            
            
            # We expect a lower nll when we do iteration...

        elif self.initial_mu == 'logm':
            
            self.mu_vec = np.log(self.m)
           
            self.mu_vec_cvx = cvx.Parameter((self.n,1),nonneg=False)
            self.mu_vec_cvx.value = self.mu_vec.reshape(-1,1)
            self.theta_mat = np.zeros((self.p, self.d))
            
            # compute the likelihood with mu = zeros and theta_hat_mat = poisson cvx
            self.theta_mat = fit_nlcv_with_cvx_v8(self.C,self.V,mu_vec = self.mu_vec)
            
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
        
    def PARALLEL_update_mu_bar_vec(self):
        eta_mat = self.V @ self.theta_mat
        eta_mat_tensor = torch.tensor(eta_mat)
        logsumexp_vec = torch.logsumexp(eta_mat_tensor, 1).detach().numpy()
        mu_star_vec = np.log(self.m) - logsumexp_vec
        
        self.mu_vec = mu_star_vec.reshape(-1,1)
        
        
    def PARALLEL_update_theta_bar_mat(self):

        start = time.perf_counter()
        words = list(np.arange(self.d))
        # Because I am using map methor of executor object,
        # the context manager below is equivalent to parallel
        # minimization of Q_kn
        with concurrent.futures.ProcessPoolExecutor() as executor:

            Cell_minQ_k=functools.partial(CELL_minQ_kn,
                                          C=self.C,
                                          V=self.V,
                                          mu_vec=self.mu_vec,
                                          verbose = False,
                                          solver = 'MOSEK') # prod_x has only one argument x (y is fixed to 10)

            result_list = executor.map(Cell_minQ_k, words)


        finish = time.perf_counter()
        print(f'Finished in {round(finish-start, 2)} second(s)')

        thtList = []
        for i in result_list:
               thtList.append(i)
                
        self.theta_mat = np.reshape(np.ravel(thtList), (self.p, self.d), order = 'F')       
        
    def PARALLEL_oneRun(self):
        ''' takes mu_vec^0 and theta_mat^0 --> updates parameters as 
        attributes of the object, mu_vec^1, theta_mat^1
        first update mu given theta,
        then update theta given mu 
        '''
        self.PARALLEL_update_mu_bar_vec()
        self.PARALLEL_update_theta_bar_mat() 
        
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
        #updates theta_matrix given mu(takes from the object attribute)
        '''Dimensions should be as follows:
        self.C = n,d
        self.V = n,p
        self.m = (n,)
        self.mu_vec = (n,)
        '''    
        self.theta_mat = fit_nlcv_with_cvx_v8(self.C,self.V,mu_vec = self.mu_vec)
    
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
        
    


class textData:
    def __init__(self,C,V,m):
        self.C= C
        self.V = V
        self.m = m


# Newest version that has polarizations and Newton-Raphson.
class MDR_v9:
    def __init__(self, textData_obj):
        # get data from textData object
        self.data = textData_obj
        self.C= textData_obj.C
        self.V = textData_obj.V
        self.m = textData_obj.m
        # put the dimensions as attributes so that I dont have to
        # define them every time.
        self.n = self.V.shape[0]
        self.p = self.V.shape[1]
        self.d = self.C.shape[1]
        
        self.mu_vec = np.zeros((self.n,1))
        
    def initialize(self):
        if self.initial_mu == 'zero': 
            print('mu initializing with zero')
            start = time.time()
            self.mu_vec = np.zeros((self.n,1))
            self.mu_vec_cvx = cvx.Parameter((self.n,1),nonneg=False)
            self.mu_vec_cvx.value = self.mu_vec
            
            self.theta_mat = np.zeros((self.p, self.d))
            # compute the likelihood with mu = zeros and theta_hat_mat = poisson cvx
            self.theta_mat = fit_nlcv_with_cvx_v8(self.C,self.V,mu_vec = self.mu_vec)
            end = time.time()
            time_took = end - start
            print("time spent in initialing 1 object (solving cvx problem)') : ",time_took)
            
            # We expect a lower nll when we do iteration...

        elif self.initial_mu == 'logm':
            
            self.mu_vec = np.log(self.m)
           
            self.mu_vec_cvx = cvx.Parameter((self.n,1),nonneg=False)
            self.mu_vec_cvx.value = self.mu_vec.reshape(-1,1)
            self.theta_mat = np.zeros((self.p, self.d))
            
            # compute the likelihood with mu = zeros and theta_hat_mat = poisson cvx
            self.theta_mat = fit_nlcv_with_cvx_v8(self.C,self.V,mu_vec = self.mu_vec)
            
        else:
            return('Error: No such start as ' + str(self.initial_mu)) 
        
    def PARALLEL_initialize(self):
        if self.initial_mu == 'zero': 
            print('mu initializing with zero')
            start = time.time()
            self.mu_vec = np.zeros((self.n,1))
            self.PARALLEL_update_theta_bar_mat()
            end = time.time()
            time_took = end - start
            print("time spent in initialing 1 object (solving 1 tour PARALLEL_update_theta_bar_mat)') : ",time_took)
        elif self.initial_mu == 'logm':
            print('mu initializing with logm')
            start = time.time()
            self.mu_vec = np.log(self.m).reshape(-1,1)
            self.PARALLEL_update_theta_bar_mat()
            end = time.time()
            time_took = end - start
            print("time spent in initialing 1 object (solving 1 tour PARALLEL_update_theta_bar_mat)') : ",time_took)
        else:
            return('Error: No such start as ' + str(self.initial_mu)) 
        
    def PARALLEL_update_mu_bar_vec(self):
        eta_mat = self.V @ self.theta_mat
        eta_mat_tensor = torch.tensor(eta_mat)
        logsumexp_vec = torch.logsumexp(eta_mat_tensor, 1).detach().numpy()
        mu_star_vec = np.log(self.m) - logsumexp_vec
        
        self.mu_vec = mu_star_vec.reshape(-1,1)
        
        
    def PARALLEL_update_theta_bar_mat(self):

        start = time.perf_counter()
        words = list(np.arange(self.d))
        # Because I am using map methor of executor object,
        # the context manager below is equivalent to parallel
        # minimization of Q_kn
        with concurrent.futures.ProcessPoolExecutor() as executor:

            Cell_minQ_k=functools.partial(CELL_minQ_kn,
                                          C=self.C,
                                          V=self.V,
                                          mu_vec=self.mu_vec,
                                          verbose = False,
                                          solver = 'MOSEK') # prod_x has only one argument x (y is fixed to 10)

            result_list = executor.map(Cell_minQ_k, words)


        finish = time.perf_counter()
        print(f'Finished in {round(finish-start, 2)} second(s)')

        thtList = []
        for i in result_list:
               thtList.append(i)
                
        self.theta_mat = np.reshape(np.ravel(thtList), (self.p, self.d), order = 'F')       
        
    def PARALLEL_oneRun(self):
        ''' takes mu_vec^0 and theta_mat^0 --> updates parameters as 
        attributes of the object, mu_vec^1, theta_mat^1
        first update mu given theta,
        then update theta given mu 
        '''
        self.PARALLEL_update_mu_bar_vec()
        self.PARALLEL_update_theta_bar_mat() 
        
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
        #updates theta_matrix given mu(takes from the object attribute)
        '''Dimensions should be as follows:
        self.C = n,d
        self.V = n,p
        self.m = (n,)
        self.mu_vec = (n,)
        '''    
        self.theta_mat = fit_nlcv_with_cvx_v8(self.C,self.V,mu_vec = self.mu_vec)
    
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
        for itera in range(num_iter):
            self.theta_mat = self.fit_NR_poisson()
        
        self.normalized_theta = normalize(self.theta_mat)
        return(self.normalized_theta,self.theta_mat)
        
        
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
        
    def compute_q_mat(self):
        # to compute q,pi etc we need to have party dummy to be the last covariate.
        self.mu_mat = np.outer(self.mu_vec, np.ones(self.d))
        eta_mat = self.V @ self.theta_mat + self.mu_mat
        lamda_mat = np.exp(eta_mat)
        rowSums = lamda_mat.sum(axis = 1)
        self.q_mat = lamda_mat/ rowSums[:,None]
        
    def compute_q_R_mat(self):
        # helper function for computing pi_vec
        VV = self.V.copy()
        VV[:,(self.p-1)] = np.ones(self.n)
        
        
        self.mu_mat = np.outer(self.mu_vec, np.ones(self.d))
        eta_mat = VV @ self.theta_mat + self.mu_mat
        lamda_mat = np.exp(eta_mat)
        rowSums = lamda_mat.sum(axis = 1)
        self.qR_mat = lamda_mat/ rowSums[:,None]
        
        
    def compute_q_D_mat(self):
        # helper function for computing pi_vec
        VV = self.V.copy()
        VV[:,(self.p-1)] = np.zeros(self.n)
        
        self.mu_mat = np.outer(self.mu_vec, np.ones(self.d))
        eta_mat = VV @ self.theta_mat + self.mu_mat
        lamda_mat = np.exp(eta_mat)
        rowSums = lamda_mat.sum(axis = 1)
        self.qD_mat = lamda_mat/ rowSums[:,None]
        
    def compute_rho_mat(self):
        # helper function for computing pi_vec
        self.compute_q_D_mat()
        self.compute_q_R_mat()
        summ = self.qD_mat + self.qR_mat
        self.rho_mat = self.qR_mat / summ
        
    def compute_pi_vec(self):
        
        # This function implements the polarization computation in GST19. More specifically,
        # for every politican, compute the counterfactual q by changing their party assignment,
        # and have a counterfactual wasserstein polarization for every politician.
        
        self.compute_rho_mat() 
        first_term  = (1/2) * np.diag(self.qR_mat @ self.rho_mat.T) 
        second_term = (1/2) * np.diag(self.qD_mat @ (np.ones((self.d,self.n)) - self.rho_mat.T))
        self.pi_vec = first_term + second_term
        return(self.pi_vec.mean())
    
    def compute_WP(self, M,  numItermax = 10000):
        # This function imitates the polarization computation in GST19 but instead of their divergence
        # we use the WD. More specifically, for every politican, compute the counterfactual q by changing 
        # their party assignment, and have a counterfactual wasserstein polarization for every politician.
        self.M = M
        self.maxm = np.max(self.M)


        self.compute_rho_mat() 
        self.cwp_vec = np.zeros(self.n)
        for i in range(self.n):
            q_ri = self.qR_mat[i,:]
            q_di = self.qD_mat[i,:]
            self.cwp_vec[i] = ot.emd2(a=q_ri, b=q_di ,M = M, numItermax=numItermax)
            res_vec = self.cwp_vec.copy()

        self.WP = res_vec.mean()
        self.std_WP = self.WP / self.maxm
        return(self.std_WP, self.WP)
    
    
    def predict_proba(self, X, mu):
        # Given a new covariate matrix, compute q matrix. with the parameters self.theta_mat, self.mu_vec
        
        mu_mat = np.outer(mu, np.ones(self.d))
        eta_mat = X @ self.theta_mat 
        #eta_mat = X @ self.theta_mat + mu_mat
        lamda_mat = np.exp(eta_mat)
        rowSums = lamda_mat.sum(axis = 1)
        qx_mat = lamda_mat/ rowSums[:,None]
        return(qx_mat)
    
    def compute_WAPP(self, M,   numItermax = 10000):
        # party indicator should be last column of V matrix
        self.M = M
        self.maxm = np.max(self.M)
        
        #slice the V into democrats and republicans
        dem_idx = ( self.V[:,(self.p - 1)] == 0 )
        rep_idx = ( self.V[:,(self.p - 1)] == 1 )

        dem_V = self.V[dem_idx].copy()
        rep_V = self.V[rep_idx].copy()
        
        
        avg_democrat_mu   = np.mean(self.mu_vec[dem_idx])
        avg_republican_mu = np.mean(self.mu_vec[rep_idx])
        
        # compute average democrats and predict q of avg dem
        self.average_democrat = np.mean(dem_V, axis = 0).reshape(1,-1)
        self.q_avg_dem = self.predict_proba(self.average_democrat, avg_democrat_mu)
        print(self.q_avg_dem)
        # compute average republican and predict q of avg rep
         
        self.average_republican = np.mean(rep_V, axis = 0).reshape(1,-1)
        self.q_avg_rep = self.predict_proba(self.average_republican, avg_republican_mu)
        print(self.q_avg_rep)
        # compute the WD between the q_avg_dem and q_avg_rep
        self.WAPP = ot.emd2(a=self.q_avg_rep.reshape(-1),
                               b=self.q_avg_dem.reshape(-1), 
                               M = M,
                               numItermax=numItermax)
        self.std_WAPP = self.WAPP / self.maxm
        return(self.std_WAPP, self.WAPP)
    
    def compute_RFWP(self, M, numItermax = 10000):
        # party indicator should be last column of V matrix
        self.M = M
        self.maxm = np.max(self.M)
        
        #slice the C into democrats and republicans
        dem_idx = ( self.V[:,(self.p - 1)] == 0 )
        rep_idx = ( self.V[:,(self.p - 1)] == 1 )
        
        dem_C = self.C[dem_idx]
        rep_C = self.C[rep_idx]
        # aggregate democrat and republican C's to get q's.
        dem_q = dem_C.mean(axis = 0)/dem_C.mean(axis = 0).sum()
        rep_q = rep_C.mean(axis = 0)/rep_C.mean(axis = 0).sum()
        
        self.RFWP = ot.emd2( a=rep_q,
                                b=dem_q, 
                                M = self.M,
                                numItermax=numItermax)


        self.std_RFWP = self.RFWP / self.maxm
        return(self.std_RFWP, self.RFWP)
 
    
    
    def estimate_mdr_with_random_parties(self, seed, initial_mu, num_epochs):
        self.backup_V = self.V.copy()

        np.random.seed(seed)
        self.V[:,(self.p-1)] = np.random.binomial(n = 1, p = 0.5,size = self.n)
        self.normalized_theta, self.theta_mat = self.fit( num_epochs = num_epochs, initial_mu = initial_mu, verbose = False)

        return(self.normalized_theta, self.theta_mat)
    
    
        
        
    def computeRandom_WP(self, seed, M, initial_mu, num_epochs, numItermax, B = 100):


        summ = 0
        for i in range(B):
            __ , _ = self.estimate_mdr_with_random_parties(seed+i, initial_mu, num_epochs ) # updates parameters.. with new V
            std_wtp, wtp = self.compute_WP(M, numItermax = numItermax)
            summ = summ + (1/B) * std_wtp # computes wtp for B different randomizations and takes the mean of them.
            self.V = self.backup_V.copy()

        return(summ)    
            
    def computeRandom_WAPP(self, seed, M, initial_mu, numItermax, num_epochs,  B = 100):
        summ = 0
        for i in range(B):
            __ ,  _ = self.estimate_mdr_with_random_parties(seed+i, initial_mu, num_epochs) # updates parameters.. with new V
            wapp, _ = self.compute_WAPP(M, numItermax = numItermax)
            summ = summ + (1/B) * wapp # computes wtp for B different randomizations and takes the mean of them.
            self.V = self.backup_V.copy()

        return(summ)    
        
        
    def computeRandom_TP(self, seed, M, B = 100):
        summ = 0
        for i in range(B):
            __ , _ = self.estimate_mdr_with_random_parties(seed+i, initial_mu = 'logm', num_epochs = 40 ) # updates parameters.. with new V
            pi = self.compute_pi_vec()
            summ = summ + (1/B) * pi # computes wapp for B different randomizations and takes the mean of them.
            self.V = self.backup_V.copy()

        return(summ)    
        
    def computeRandom_RFWP(self, seed, M, numItermax = 10000, B=100, initial_mu = 'logm', num_epochs = 1):
        summ = 0
        for i in range(B):
            __ ,  _ = self.estimate_mdr_with_random_parties(seed+i, initial_mu, num_epochs) # updates parameters.. with new V
            rfwp, _ = self.compute_RFWP(M, numItermax = numItermax)
            summ = summ + (1/B) * rfwp # computes wtp for B different randomizations and takes the mean of them.
            self.V = self.backup_V.copy()
        return(summ)    

        
# print('Imported functions module')


############################################################################################################################################
##########################################                 FUNCTIONS                      ##################################################
############################################################################################################################################


def set_hyperparams(n = 1000, d = 3, p = 5, V_mean = 0, V_std = 1, V_dist = 'normal'):
    param_dict = {'n' : n,
                  'd' : d, 
                  'p' : p,
                  'V_mean' : V_mean,
                  'V_std' : V_std,
                  'V_dist' : V_dist
        }
    return(n,d,p,V_mean,V_std,V_dist,param_dict)


def set_true_params(seed, param_dict):
    np.random.seed(seed)
    thetaTRUE_mat= np.random.normal(param_dict['V_mean'],param_dict['V_std'],(param_dict['p'],param_dict['d']))
    muTRUE_vec=np.random.normal(param_dict['V_mean'],param_dict['V_std'], param_dict['n'])
    #muTRUE_vec = np.zeros((param_dict['n'],1))
    return(seed,thetaTRUE_mat, muTRUE_vec)

## Function cell
def set_hyperparams(n = 1000, d = 3, p = 5, V_mean = 0, V_std = 1, V_dist = 'normal'):
    if V_dist =="mixture":
        # V_mean2 and V_std2 and fraction of covariates are fixed, not user-defined. Gotta be fixed in an ideal world.
        param_dict = {'n' : n,
                  'd' : d, 
                  'p' : p,
                  'V_mean' : V_mean,
                  'V_std' : V_std,
                  'V_mean1' : V_mean,
                  'V_std1' : V_std,
                  'V_mean2' : V_mean+4,
                  'V_std2' : V_std,
                  'fraction' : 0.5
        }
        return(n,d,p,V_mean,V_std,V_dist,param_dict)
        
        
    param_dict = {'n' : n,
                  'd' : d, 
                  'p' : p,
                  'V_mean' : V_mean,
                  'V_std' : V_std,
                  'V_dist' : V_dist
        }
    return(n,d,p,V_mean,V_std,V_dist,param_dict)

def sample_mixture_of_normals(mean1, std_dev1, mean2, std_dev2, shape=(1,)):
    """
    Returns a matrix of samples drawn from a mixture of two 
    normal distributions with specified means and standard deviations.
    
    Parameters:
    mean1 (float): Mean of the first normal distribution.
    std_dev1 (float): Standard deviation of the first normal distribution.
    mean2 (float): Mean of the second normal distribution.
    std_dev2 (float): Standard deviation of the second normal distribution.
    shape (tuple): The shape of the output matrix. Default is (1,).
    
    Returns:
    numpy.ndarray: A matrix of samples drawn from the mixture of the two normal distributions.
    """
    # calculate total number of samples
    num_samples = np.prod(shape)
    
    # choose the distribution from which to draw each sample
    choices = np.random.choice([0, 1], size=num_samples)
    
    # draw the samples
    samples = [np.random.normal(loc=[mean1, mean2][i], scale=[std_dev1, std_dev2][i]) for i in choices]
    
    # reshape the samples into a matrix
    samples_matrix = np.reshape(samples, shape)
    
    return samples_matrix
def get_CVM_from_mixture_mdl(thetaTRUE_mat, seed, param_dict):
        
        n = param_dict['n']
        d = param_dict['d']
        p = param_dict['p']
        V_mean = param_dict['V_mean']
        V_std = param_dict['V_std']
        V_mean1 = param_dict['V_mean1']
        V_std1 = param_dict['V_std1']
        V_mean2 = param_dict['V_mean2']
        V_std2 = param_dict['V_std2']
        fraction = param_dict['fraction']

        # this generates covariate matrix V
        mixture_cov_num = int(p/2)
        normal_cov_num = p - mixture_cov_num
        
        np.random.seed(seed)
        low_V = np.random.normal(V_mean, V_std, (n,normal_cov_num))
        np.random.seed(seed)
        high_V = sample_mixture_of_normals(V_mean1, V_std1, V_mean2, V_std2, (n,mixture_cov_num))
        V = np.hstack((low_V, high_V))

        
        
        #Generate m_vec.
        m = sample_mixture_of_normals(10, 1, 60, 5, n)
        intMaker = lambda t: int(t)
        m = np.array([intMaker(np.abs(mi)) for mi in m])
        
        
        eta_mat = ( V @ thetaTRUE_mat ) 
        lamda_mat = np.exp(eta_mat)
        rowSums =  lamda_mat.sum(axis =1)
        q_mat = lamda_mat / ( np.repeat(rowSums , d , axis = 0 ).reshape(n, d) )

        #Fill C_matrix by sampling from MNL
        C = np.zeros((n,d))
        for i in range(n): # this loop fills C matrix
                np.random.seed(seed + i)
                C[i,] = np.random.multinomial(m[i], q_mat[i,], size=1)

        return(C,V,m)

def get_CVM_from_Poisson_mdl(muTRUE_vec, thetaTRUE_mat, n, seed, V_mean =0 , V_std = 1):
        
        p = thetaTRUE_mat.shape[0]
        d = thetaTRUE_mat.shape[1]
        
        np.random.seed(seed)
        V = np.random.normal(V_mean, V_std, (n,p)) # this generates covariate matrix V

        mu_repeated_mat  = muTRUE_vec.reshape(-1,1) @ np.ones((1,d))
        
        eta_mat = ( V @ thetaTRUE_mat ) + mu_repeated_mat

        lamda_mat = np.exp(eta_mat)# this  fills C matrix
        

        C = np.zeros((n,d))
        for j in range(d):
            C[:,j] = np.random.poisson(lam = lamda_mat[:,j], size = n)

        m = C.sum(axis =1)  # this assigns m_vec 
        
        zero_m_bool = (m != 0)
        C = C[zero_m_bool]
        V = V[zero_m_bool]
        m = m[zero_m_bool]
        muTRUE_vec = muTRUE_vec[zero_m_bool]
        
        return(C,V,m, muTRUE_vec)

# def get_CVM_from_MNL_mdl(thetaTRUE_mat, n, seed, lower = 20,upper = 30, V_mean = 0, V_std = 1):
#         '''
#          does not use muTRUE_vec, can be inputted as None
#          Updates C,V,m of the fakeDataGenerator object from Multinomial Model.'''
#         p = thetaTRUE_mat.shape[0]
#         d = thetaTRUE_mat.shape[1]
        
#         np.random.seed(seed)
#         V = np.random.normal(V_mean, V_std, (n,p)) # this generates covariate matrix V
#         #Generate m_vec.
#         m = np.round(np.random.uniform(lower,upper,n))
#         intMaker = lambda t: int(t)
#         m = np.array([intMaker(mi) for mi in m])
        
#         #compute q matrix(or q_i for all i)
#         #err_mat = np.random.gumbel(loc=0.0, scale=1.0, size=(n,d))
#         #eta_mat = ( V @ thetaTRUE_mat ) + err_mat
#         eta_mat = ( V @ thetaTRUE_mat ) 
        
#         lamda_mat = np.exp(eta_mat)
#         rowSums =  lamda_mat.sum(axis =1)
#         q_mat = lamda_mat / ( np.repeat(rowSums , d , axis = 0 ).reshape(n, d) )

#         #Fill C_matrix by sampling from MNL
#         C = np.zeros((n,d))
#         for i in range(n): # this loop fills C matrix
#                 np.random.seed(seed + i)
#                 C[i,] = np.random.multinomial(m[i], q_mat[i,], size=1)

#         return(C,V,m)

def get_CVM_from_MNL_mdl(thetaTRUE_mat, n, seed, lower = 20,upper = 30, V_mean = 0, V_std = 1):
        '''
         does not use muTRUE_vec, can be inputted as None
         Updates C,V,m of the fakeDataGenerator object from Multinomial Model.'''
        p = thetaTRUE_mat.shape[0]
        d = thetaTRUE_mat.shape[1]
        
        np.random.seed(seed)
        
        # with constant column.

        V = np.random.normal(V_mean, V_std, (n,p-1)) # this generates covariate matrix V
        constant_column = np.ones((V.shape[0], 1), dtype=V.dtype)
        V = np.hstack((constant_column, V))


        #Generate m_vec.
        m = np.round(np.random.uniform(lower,upper,n))
        intMaker = lambda t: int(t)
        m = np.array([intMaker(mi) for mi in m])
        
        #compute q matrix(or q_i for all i)
        #err_mat = np.random.gumbel(loc=0.0, scale=1.0, size=(n,d))
        #eta_mat = ( V @ thetaTRUE_mat ) + err_mat
        eta_mat = ( V @ thetaTRUE_mat ) 
        
        lamda_mat = np.exp(eta_mat)
        rowSums =  lamda_mat.sum(axis =1)
        q_mat = lamda_mat / ( np.repeat(rowSums , d , axis = 0 ).reshape(n, d) )

        #Fill C_matrix by sampling from MNL
        C = np.zeros((n,d))
        for i in range(n): # this loop fills C matrix
                np.random.seed(seed + i)
                C[i,] = np.random.multinomial(m[i], q_mat[i,], size=1)

        return(C,V,m)

def generate_DATA_4BS(theta_hat, seed, n, d, p, model):
    if model == 'MNL':   
        print('Generating data from MNL model.. with the theta_hat specified for the parametric bootstrap')

        n,d,p,V_mean,V_std,V_dist,param_dict = set_hyperparams(n = n, 
                                                    d = d,
                                                    p = p,
                                                    V_mean = 0,
                                                    V_std = 1, 
                                                    V_dist = 'normal'
                                                    )
        C,V,m = get_CVM_from_MNL_mdl(theta_hat, n, seed, lower = 20,upper = 30)

        data_hat = textData(C,V,m)
    else:
        print('parametric bootstrap currently only supports when DGP is MNL')
        return(None)
    return(data_hat)

def plot_element_distribution(bootstrap_theta_hats, element_index):
    """
    Plots the distribution of the specified element across different matrices.

    :param bootstrap_theta_hats: List of matrices.
    :param element_index: Tuple (i, j) for the row and column index.
    """
    values = [theta_hat[element_index[0], element_index[1]] for theta_hat in bootstrap_theta_hats]

    # Creating the scatter plot
    plt.scatter(values, alpha=0.5)
    plt.title(f'Distribution of Element at Index {element_index}')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.show()


    
def generate_DATA(seed, n, d, p ,model = 'Po', seed_theta = 1233):

    if model =='mixture':
        print('Generating data from Mixture MNL..')
        np.random.seed(seed_theta)
        thetaTRUE_mat= np.random.normal(0,1,(p,d))
        n,d,p,V_mean,V_std,V_dist,param_dict = set_hyperparams(n = n, 
                                                    d = d,
                                                    p = p,
                                                    V_mean = 0,
                                                    V_std = 1, 
                                                    V_dist = 'mixture'
                                                    )
        

        C,V,m = get_CVM_from_mixture_mdl(thetaTRUE_mat, seed, param_dict)
        data = textData(C,V,m)
        return(data, thetaTRUE_mat)
    
        
    if model == 'Po':   
        # generate data from Poisson model with muTRUE = 0
        print('Generating data from Poisson..')
        # Set hyper params

        n,d,p,V_mean,V_std,V_dist,param_dict = set_hyperparams(n = n, 
                                                    d = d,
                                                    p = p,
                                                    V_mean = 0,
                                                    V_std = 1, 
                                                    V_dist = 'normal'
                                                    )

        np.random.seed(seed_theta)
        thetaTRUE_mat= np.random.normal(0,1,(p,d))
    
        #muTRUE_vec=np.random.normal(0,1, n)
        #muTRUE_vec=np.ones(n)
        muTRUE_vec=np.zeros(n)


        C,V,m,muTRUE_vec = get_CVM_from_Poisson_mdl(muTRUE_vec, thetaTRUE_mat, n, seed)

        data = textData(C,V,m)
        return(data,muTRUE_vec, thetaTRUE_mat)
    
    if model == 'MNL':   
        # generate data from Poisson model with muTRUE = 0
        print('Generating data from MNL model..')
        # Set hyper params

        n,d,p,V_mean,V_std,V_dist,param_dict = set_hyperparams(n = n, 
                                                    d = d,
                                                    p = p,
                                                    V_mean = 0,
                                                    V_std = 1, 
                                                    V_dist = 'normal'
                                                    )

        np.random.seed(seed_theta)
        thetaTRUE_mat= np.random.normal(0,1,(p,d))

        C,V,m = get_CVM_from_MNL_mdl(thetaTRUE_mat, n, seed, lower = 20,upper = 30)

        data = textData(C,V,m)
        return(data, thetaTRUE_mat)
    
def compute_nll(theta_mat, mu_vecc, V,C):
    sum_j = 0
    d = theta_mat.shape[1]
    for j in range(d):
        eta_vec = V @ theta_mat[:,j] + mu_vecc
        lamda_vec = np.exp(eta_vec)
        res_col = (lamda_vec - ( C[:,j] * eta_vec) ).mean()
        sum_j += res_col
    return(sum_j)


def average_list(maList):
    size = len(maList)
    toplam =0
    for i in range(size):
        toplam += maList[i]
    return(toplam/size)
    


def compute_nll_l_cv(theta_mat, mu_vec, C, V):
    # PROBABLY HAVE A MISTAKE HERE
    n = mu_vec.shape[0]
    p = theta_mat.shape[0]
    d = theta_mat.shape[1]
    
    mu_mat = mu_vec.reshape((n,1)) @ np.ones((1,d))
    
    eta_mat = V @ theta_mat 
    eta_bar_mat = eta_mat + mu_mat
    
    res_mat = np.exp(eta_bar_mat) - C*eta_bar_mat
    NLCV = np.sum(res_mat)
    
    return(NLCV)

# =
def compute_nll_l_cvm(theta_mat, mu_vec, C, V, m):
    # PROBABLY HAVE A MISTAKE HERE
    n = mu_vec.shape[0]
    p = theta_mat.shape[0]
    d = theta_mat.shape[1]
    
    mu_mat = mu_vec.reshape((n,1)) @ np.ones((1,d))
    
    eta_mat = V @ theta_mat 
    eta_bar_mat = eta_mat + mu_mat
    eta_bar_vec = eta_bar_mat.sum(axis = 1)
    
    lse_etabar_vec = scipy.special.logsumexp(eta_bar_mat, axis = 1)
    nlcvm = np.dot(m,lse_etabar_vec) -  np.diag(C@eta_bar_mat.T).sum()

    return(nlcvm)
# +
def compute_nll_l_mv(theta_mat, mu_vec, C, V, m):
    n = mu_vec.shape[0]
    p = theta_mat.shape[0]
    d = theta_mat.shape[1]
    
    mu_mat = mu_vec.reshape((n,1)) @ np.ones((1,d))
    
    eta_mat = V @ theta_mat 
    eta_bar_mat = eta_mat + mu_mat
    eta_bar_vec = eta_bar_mat.sum(axis = 1)
    
    lse_etabar_vec = scipy.special.logsumexp(eta_bar_mat, axis = 1)
    
    m_log_factorial_vec = scipy.special.gammaln(m+1)

    nlmv = -1* np.dot(m,lse_etabar_vec) + np.sum(np.exp(eta_bar_mat)) 

    return(nlmv)

# need to change the FDGP to add party loadings and increase d to create bias in WD estimator.

def get_congressional_CVM_from_Poisson_mdl(muTRUE_vec, thetaTRUE_mat, n, seed, V_mean =0 , V_std = 1):
        
        p = thetaTRUE_mat.shape[0]
        d = thetaTRUE_mat.shape[1]
        
        np.random.seed(seed)
        V = np.random.normal(V_mean, V_std, (n,p)) # this generates covariate matrix V
        V[:,(p-1)] = np.random.binomial(n = 1, p = 0.5,size = n)
        mu_repeated_mat  = muTRUE_vec.reshape(-1,1) @ np.ones((1,d))
        
        eta_mat = ( V @ thetaTRUE_mat ) + mu_repeated_mat

        lamda_mat = np.exp(eta_mat)# this  fills C matrix
        

        
        C = np.zeros((n,d))
        for j in range(d):
            C[:,j] = np.random.poisson(lam = lamda_mat[:,j], size = n)

        m = C.sum(axis =1)  # this assigns m_vec 
        
        zero_m_bool = (m != 0)
        C = C[zero_m_bool]
        V = V[zero_m_bool]
        m = m[zero_m_bool]
        muTRUE_vec = muTRUE_vec[zero_m_bool]
        
        return(C,V,m, muTRUE_vec)

def get_congressional_CVM_from_MNL_mdl(thetaTRUE_mat, n, seed, lower = 20,upper = 30, V_mean = 0, V_std = 1):
        '''
         does not use muTRUE_vec, can be inputted as None
         Updates C,V,m of the fakeDataGenerator object from Multinomial Model.'''
        p = thetaTRUE_mat.shape[0]
        d = thetaTRUE_mat.shape[1]
        
        np.random.seed(seed)
        #DUZELT
        #V = np.random.normal(V_mean, V_std, (n,p)) # this generates covariate matrix V

        V = np.random.chisquare(2,(n,p))
        

        V[:,(p-1)] = np.random.binomial(n = 1, p = 0.5,size = n)
        #Generate m_vec.
        m = np.round(np.random.uniform(lower,upper,n))
        intMaker = lambda t: int(t)
        m = np.array([intMaker(mi) for mi in m])
        
        eta_mat = ( V @ thetaTRUE_mat ) 
        
        lamda_mat = np.exp(eta_mat)
        rowSums =  lamda_mat.sum(axis =1)
        q_mat = lamda_mat / ( np.repeat(rowSums , d , axis = 0 ).reshape(n, d) )

        #Fill C_matrix by sampling from MNL
        C = np.zeros((n,d))
        for i in range(n): # this loop fills C matrix
                np.random.seed(seed + i)
                C[i,] = np.random.multinomial(m[i], q_mat[i,], size=1)

        return(C,V,m)
    
def generate_congressional_DATA(seed, n, d, p ,model = 'Po', seed_theta = 1233, strength_party = 1, const = True  ):

    if strength_party == 999 and model == 'MNL':
        print('Generating data with disjoint support with d = 3 from MNL model..')
        # Set hyper params

        n,d,p,V_mean,V_std,V_dist,param_dict = set_hyperparams(n = n, 
                                                    d = 3,
                                                    p = p,
                                                    V_mean = 0,
                                                    V_std = 1, 
                                                    V_dist = 'normal'
                                                    )

        np.random.seed(seed_theta)
        thetaTRUE_mat= np.random.normal(0,1,(p,d))
        thetaTRUE_mat[p-1,:] = np.array([5,3,0])
        thetaTRUE_mat[p-2,:] = np.array([0,3,5])
        # modifying thetaTRUE_mat so that it makes sense with a certain M.
        # party loading parameters for word pairs that have high M values should be diff,
        # so that we can observe a meaningful difference between parties.
        # I am not modufying because with these seed settings the M and thetaTRUE_mat makes sense.
        #i.e. for different params in party loadings for different words, we see high M.

        C,V,m = get_congressional_HARM_CVM_from_MNL_mdl(thetaTRUE_mat, n, seed, lower = 20,upper = 30)

        if const == True: 
            V[:,0] = np.ones(n)

        data = textData(C,V,m)
        return(data, thetaTRUE_mat)


    if model == 'Po':   
        # generate data from Poisson model with muTRUE = 0
        print('Generating data from Poisson..')
        # Set hyper params

        n,d,p,V_mean,V_std,V_dist,param_dict = set_hyperparams(n = n, 
                                                    d = d,
                                                    p = p,
                                                    V_mean = 0,
                                                    V_std = 1, 
                                                    V_dist = 'normal'
                                                    )

        np.random.seed(seed_theta)
        thetaTRUE_mat= np.random.normal(0,1,(p,d))
        thetaTRUE_mat[p-1,:] = thetaTRUE_mat[p-1,:]*strength_party
        
        muTRUE_vec=np.zeros(n)


        C,V,m,muTRUE_vec = get_congressional_CVM_from_Poisson_mdl(muTRUE_vec,
                                                                  thetaTRUE_mat,
                                                                  n,
                                                                  seed+1)


        if const == True: 
            V[:,0] = np.ones(n)

        data = textData(C,V,m)
        return(data,muTRUE_vec, thetaTRUE_mat)
    
    if model == 'MNL':   
        
        print('Generating data from MNL model..')
        # Set hyper params

        n,d,p,V_mean,V_std,V_dist,param_dict = set_hyperparams(n = n, 
                                                    d = d,
                                                    p = p,
                                                    V_mean = 0,
                                                    V_std = 1, 
                                                    V_dist = 'normal'
                                                    )

        np.random.seed(seed_theta)
        thetaTRUE_mat= np.random.normal(0,1,(p,d))
        thetaTRUE_mat[p-1,:] = np.random.uniform(-2,2,d)
        # modifying thetaTRUE_mat so that it makes sense with a certain M.
        # party loading parameters for word pairs that have high M values should be diff,
        # so that we can observe a meaningful difference between parties.
        # I am not modufying because with these seed settings the M and thetaTRUE_mat makes sense.
        #i.e. for different params in party loadings for different words, we see high M.

        C,V,m = get_congressional_CVM_from_MNL_mdl(thetaTRUE_mat, n, seed, lower = 20,upper = 30)

        if const == True: 
            V[:,0] = np.ones(n)

        data = textData(C,V,m)
        return(data, thetaTRUE_mat)



def get_congressional_HARM_CVM_from_MNL_mdl(thetaTRUE_mat, n, seed, lower = 20,upper = 30, V_mean = 0, V_std = 1):
        '''
         does not use muTRUE_vec, can be inputted as None
         Updates C,V,m of the fakeDataGenerator object from Multinomial Model.'''
        p = thetaTRUE_mat.shape[0]
        d = thetaTRUE_mat.shape[1]
        
        np.random.seed(seed)
        V = np.random.normal(V_mean, V_std, (n,p)) # this generates covariate matrix V
        V[:,(p-1)] = np.random.binomial(n = 1, p = 0.5,size = n)
        V[:,(p-2)] = np.abs(V[:,(p-1)] -1)
        #Generate m_vec.
        m = np.round(np.random.uniform(lower,upper,n))
        intMaker = lambda t: int(t)
        m = np.array([intMaker(mi) for mi in m])
        
        #compute q matrix(or q_i for all i)
        #err_mat = np.random.gumbel(loc=0.0, scale=1.0, size=(n,d))
        #eta_mat = ( V @ thetaTRUE_mat ) + err_mat
        eta_mat = ( V @ thetaTRUE_mat ) 
        
        lamda_mat = np.exp(eta_mat)
        rowSums =  lamda_mat.sum(axis =1)
        q_mat = lamda_mat / ( np.repeat(rowSums , d , axis = 0 ).reshape(n, d) )

        #Fill C_matrix by sampling from MNL
        C = np.zeros((n,d))
        for i in range(n): # this loop fills C matrix
                np.random.seed(seed + i)
                C[i,] = np.random.multinomial(m[i], q_mat[i,], size=1)

        return(C,V,m)



def random_party_assignment_WP(data_, M,assignment_seed = 123):
    n = data_.V.shape[0]
    p = data_.V.shape[1]
    np.random.seed(assignment_seed)
    random_assignment = np.random.binomial(1,0.5,n)
    rand_dem_idx = (random_assignment == 0)
    rand_rep_idx = (random_assignment == 1)
    
    data_.V[:,p-1] = random_assignment
    
    mdl = MDR_v9(textData_obj= data_)
    normalized_theta_hat, theta_hat = mdl.fit(num_epochs= 40,
                                              initial_mu = 'logm',
                                              verbose = False)

    normalized_wd, wd = mdl.compute_WP(M)
    
    return(normalized_wd, wd)


def random_party_assignment_WAPP(data_, M,assignment_seed = 123):
    n = data_.V.shape[0]
    p = data_.V.shape[1]
    np.random.seed(assignment_seed)
    random_assignment = np.random.binomial(1,0.5,n)
    rand_dem_idx = (random_assignment == 0)
    rand_rep_idx = (random_assignment == 1)
    
    data_randdem = textData(data_.C[rand_dem_idx],data_.V[rand_dem_idx,],data_.m[rand_dem_idx])
    data_randrep = textData(data_.C[rand_rep_idx],data_.V[rand_rep_idx,],data_.m[rand_rep_idx])

    mdlrD = MDR_v9(textData_obj= data_randdem)
    normalized_theta_hatrD, theta_hatrD = mdlrD.fit(num_epochs= 40,
                                              initial_mu = 'logm',
                                              verbose = False)
    a = mdlrD.predict_proba(mdlrD.V.mean(axis=0),0)

    mdlrR = MDR_v9(textData_obj= data_randrep)
    normalized_theta_hatrR, theta_hatrR = mdlrR.fit(num_epochs= 40,
                                              initial_mu = 'logm',
                                              verbose = False)
    b = mdlrR.predict_proba(mdlrR.V.mean(axis=0),0)


    res = ot.emd2(a=a.reshape(-1), b=b.reshape(-1), M = M, numItermax=10000)/np.max(M)
    return(res, a, b)


















