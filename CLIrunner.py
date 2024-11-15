
import multiprocessing as mp
import time
import concurrent.futures
import functools
from tqdm import tqdm
from classesv2 import *
# Command line inputting
import sys
import os 

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
    # slurm_env_var = sys.argv[10]
    # slurm_env_var = int(slurm_env_var)
    # print(slurm_env_var)

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
    b_idx_theta_list = [] 
    sq_diff = []
    abs_diff = []
    _diff = []
    time_list = []
    mse_list = []  # New list to store MSE for each iteration


    #random_start_seed = np.random.randint(10000, size = 1)
    random_start_seed = seed


    if estimator == 'ours':
        for b in range(B):
            if model == 'Po':

                data_, muTRUE_vec, thetaTRUE_mat = generate_DATA(seed = random_start_seed+b, n = n, d = d, p = p, model = model, seed_theta = seed_theta)

                normalized_thetaTRUE_mat = normalize(thetaTRUE_mat)

                PARALLEL_mdl = MDR_v9(textData_obj= data_)

                # PARALLEL_normalized_theta_hat, PARALLEL_theta_hat = PARALLEL_mdl.fit(num_epochs= num_epochs,
                #                                                                      initial_mu = initial_mu,
                #                                                                      verbose = False)

                try:
                    start = time.time()
                    PARALLEL_normalized_theta_hat, PARALLEL_theta_hat = PARALLEL_mdl.fit(num_epochs= num_epochs,
                                                                                        initial_mu = initial_mu,
                                                                                        verbose = False)
                    end = time.time()
                    time_took = end - start
                    #print("time spent ",time_took)

                except:
                    print("Oops!", sys.exc_info()[0], "occurred.")
                    print("Skipping this seed.")
                    continue

            if model == 'mixture':

                data_, thetaTRUE_mat = generate_DATA(seed = random_start_seed+b, n = n, d = d, p = p, model = 'mixture', seed_theta = seed_theta)

                normalized_thetaTRUE_mat = normalize(thetaTRUE_mat)

                PARALLEL_mdl = MDR_v9(textData_obj= data_)

                # PARALLEL_normalized_theta_hat, PARALLEL_theta_hat = PARALLEL_mdl.fit(num_epochs= num_epochs,
                #                                                                      initial_mu = initial_mu,
                #                                                                      verbose = False)

                try:
                    start = time.time()
                    PARALLEL_normalized_theta_hat, PARALLEL_theta_hat = PARALLEL_mdl.fit(num_epochs= num_epochs,
                                                                                        initial_mu = initial_mu,
                                                                                        verbose = False)
                    end = time.time()
                    time_took = end - start
                    #print("time spent ",time_took)

                except:
                    print("Oops!", sys.exc_info()[0], "occurred.")
                    print("Skipping this seed.")
                    continue
            else:
            
                data_, thetaTRUE_mat = generate_DATA(seed =random_start_seed+b, n = n, d = d, p = p, model = model, seed_theta= seed_theta)
                normalized_thetaTRUE_mat = normalize(thetaTRUE_mat)

                PARALLEL_mdl = MDR_v9(textData_obj= data_)
                PARALLEL_mdl.initial_mu = initial_mu 
                PARALLEL_mdl.mu_vec = np.log(PARALLEL_mdl.m).reshape(-1,1)
                words = list(np.arange(PARALLEL_mdl.d))
                try:
                    start = time.time()
                    PARALLEL_normalized_theta_hat, PARALLEL_theta_hat = PARALLEL_mdl.PARALLEL_fit(num_epochs= num_epochs,
                                                                                                initial_mu = initial_mu,
                                                                                                verbose = False)
                    end = time.time()
                    time_took = end - start
                    print("time spent in 1 Bootstrap sample, (DGP model='MNL') : ",time_took)

                except:
                    print("Oops!", sys.exc_info()[0], "occurred.")
                    print("Skipping this seed.")
                    continue
        
            b_idx_theta_list.append(PARALLEL_normalized_theta_hat)
            sq_diff.append( (PARALLEL_normalized_theta_hat - normalized_thetaTRUE_mat)**2 )
            abs_diff.append(  np.average(np.abs(PARALLEL_normalized_theta_hat - normalized_thetaTRUE_mat) ))
            _diff.append(  np.average(PARALLEL_normalized_theta_hat - normalized_thetaTRUE_mat))
            time_list.append(time_took)        
    if estimator == 'MLE':
        for b in tqdm(range(B), desc="Processing numerical experiments B times"):
            if model == 'Po':
                data_, muTRUE_vec, thetaTRUE_mat = generate_DATA(seed = random_start_seed+b, n = n, d = d, p = p, model = model, seed_theta = seed_theta)

                normalized_thetaTRUE_mat = normalize(thetaTRUE_mat)

                PARALLEL_mdl = MDR_v9(textData_obj= data_)

            
                try:
                    start = time.time()

                    mdl = nll_l_cvm(data_)
                    optimizer = torch.optim.AdamW(mdl.parameters(), lr=1e-4)

                    for steps in range(10000):  
                        loss = mdl() 
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        
                    PARALLEL_normalized_theta_hat = normalize(mdl.theta_layer.weight.transpose(1,0).detach().numpy())

                    end = time.time()
                    time_took = end - start
                    #print("time spent ",time_took)

                except:
                    print("Oops!", sys.exc_info()[0], "occurred.")
                    print("Skipping this seed.")
                    continue
            
            


            else:
                
                data_, thetaTRUE_mat = generate_DATA(seed =random_start_seed+b, n = n, d = d, p = p, model = model, seed_theta= seed_theta)
                normalized_thetaTRUE_mat = normalize(thetaTRUE_mat)

                PARALLEL_mdl = MDR_v9(textData_obj= data_)
                PARALLEL_mdl.initial_mu = initial_mu 
                PARALLEL_mdl.mu_vec = np.log(PARALLEL_mdl.m).reshape(-1,1)
                words = list(np.arange(PARALLEL_mdl.d))
                try:
                    start = time.time()
                    
                    mdl = nll_l_cvm(data_)
                    optimizer = torch.optim.AdamW(mdl.parameters(), lr=1e-4)
                    total = 30000
                    for steps in range(total):  
                        if steps % (total // 10) == 0:
                            print(f"Progress: {steps / total * 100:.1f}%") 
                        loss = mdl() 
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        
                    PARALLEL_normalized_theta_hat = normalize(mdl.theta_layer.weight.transpose(1,0).detach().numpy())

                    end = time.time()
                    time_took = end - start
                    #print("time spent ",time_took)

                except:
                    print("Oops!", sys.exc_info()[0], "occurred.")
                    print("Skipping this seed.")
                    continue
            
            b_idx_theta_list.append(PARALLEL_normalized_theta_hat)
            sq_diff.append( (PARALLEL_normalized_theta_hat - normalized_thetaTRUE_mat)**2 )
            abs_diff.append(  np.average(np.abs(PARALLEL_normalized_theta_hat - normalized_thetaTRUE_mat) ))
            _diff.append(  np.average(PARALLEL_normalized_theta_hat - normalized_thetaTRUE_mat))
            time_list.append(time_took)

            # Calculate and store MSE for this iteration
            current_sq_diff = (PARALLEL_normalized_theta_hat - normalized_thetaTRUE_mat)**2
            current_mse = np.mean(current_sq_diff)
            mse_list.append(current_mse)

            # Print current MSE
            print(f"Iteration {b+1}/{B} - MSE: {current_mse:.6f}")


    if estimator == 'Bohning':
        for b in range(B):
            if model == 'Po':
                data_, muTRUE_vec, thetaTRUE_mat = generate_DATA(seed = random_start_seed+b, n = n, d = d, p = p, model = model, seed_theta = seed_theta)

                normalized_thetaTRUE_mat = normalize(thetaTRUE_mat)

                try:
                    start = time.time()
                    # MLE through Bohning

                    mdl = nll_l_cvm(data_)
                    optimizer = torch.optim.AdamW(mdl.parameters(), lr=1e-1)

                    XtXinv = np.linalg.inv(np.transpose(mdl.V) @ mdl.V) #(p,p)
                    E = np.identity(d) #(d,d)
                    B_inv = 2 * np.kron( E + np.ones((d, d)), XtXinv) #(dp,dp)

                    for steps in range(1000):  
                        loss = mdl() 
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        # get the gradient:
                        fd = mdl.theta_layer.weight.grad #(d,p)
                        fd = fd.reshape(-1) #(dp,) # order of reshape could be f.ing up
                        
                        # Change the following line to Bohning92 
                        # optimizer.step()
                        params = mdl.theta_layer.weight.reshape(-1).detach().numpy()
                        params = params - B_inv @ fd.numpy()
                        with torch.no_grad():
                            mdl.theta_layer.weight = torch.nn.parameter.Parameter(torch.from_numpy(params.reshape(d,p)))
                        
                    PARALLEL_normalized_theta_hat = normalize(mdl.theta_layer.weight.transpose(1,0).detach().numpy())

                    end = time.time()
                    time_took = end - start
                    #print("time spent ",time_took)

                except:
                    print("Oops!", sys.exc_info()[0], "occurred.")
                    print("Skipping this seed.")
                    continue
            
            


            else:
                
                data_, thetaTRUE_mat = generate_DATA(seed =random_start_seed+b, n = n, d = d, p = p, model = model, seed_theta= seed_theta)
                normalized_thetaTRUE_mat = normalize(thetaTRUE_mat)
                
                try:
                    start = time.time()
                    # MLE through Bohning

                    mdl = nll_l_cvm(data_)
                    optimizer = torch.optim.AdamW(mdl.parameters(), lr=1e-1)

                    XtXinv = np.linalg.inv(np.transpose(mdl.V) @ mdl.V) #(p,p)
                    E = np.identity(d) #(d,d)
                    B_inv = 2 * np.kron( E + np.ones((d, d)), XtXinv) #(dp,dp)

                    for steps in range(1000):  
                        loss = mdl() 
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        # get the gradient:
                        fd = mdl.theta_layer.weight.grad #(d,p)
                        fd = fd.reshape(-1) #(dp,) # order of reshape could be f.ing up
                        
                        # Change the following line to Bohning92 
                        # optimizer.step()
                        params = mdl.theta_layer.weight.reshape(-1).detach().numpy()
                        params = params - B_inv @ fd.numpy()
                        with torch.no_grad():
                            mdl.theta_layer.weight = torch.nn.parameter.Parameter(torch.from_numpy(params.reshape(d,p)))
                        
                    PARALLEL_normalized_theta_hat = normalize(mdl.theta_layer.weight.transpose(1,0).detach().numpy())

                    end = time.time()
                    time_took = end - start
                    #print("time spent ",time_took)

                except:
                    print("Oops!", sys.exc_info()[0], "occurred.")
                    print("Skipping this seed.")
                    continue
            
            b_idx_theta_list.append(PARALLEL_normalized_theta_hat)
            sq_diff.append( (PARALLEL_normalized_theta_hat - normalized_thetaTRUE_mat)**2 )
            abs_diff.append(  np.average(np.abs(PARALLEL_normalized_theta_hat - normalized_thetaTRUE_mat) ))
            _diff.append(  np.average(PARALLEL_normalized_theta_hat - normalized_thetaTRUE_mat))
            time_list.append(time_took)
    if estimator == 'NR':
        for b in range(B):
            if model == 'Po':
                data_, muTRUE_vec, thetaTRUE_mat = generate_DATA(seed = random_start_seed+b, n = n, d = d, p = p, model = model, seed_theta = seed_theta)

                normalized_thetaTRUE_mat = normalize(thetaTRUE_mat)

                PARALLEL_mdl = MDR_v9(textData_obj= data_)


                try:
                    start = time.time()
                    PARALLEL_normalized_theta_hat, PARALLEL_theta_hat = PARALLEL_mdl.fit_NewtonRaphson( num_iter= 100)
                    end = time.time()
                    time_took = end - start
                    #print("time spent ",time_took)

                except:
                    print("Oops!", sys.exc_info()[0], "occurred.")
                    print("Skipping this seed.")
                    continue
            
            


            else:
                
                data_, thetaTRUE_mat = generate_DATA(seed =random_start_seed+b, n = n, d = d, p = p, model = model, seed_theta= seed_theta)
                normalized_thetaTRUE_mat = normalize(thetaTRUE_mat)

                PARALLEL_mdl = MDR_v9(textData_obj= data_)
                PARALLEL_mdl.initial_mu = initial_mu 
                PARALLEL_mdl.mu_vec = np.log(PARALLEL_mdl.m).reshape(-1,1)
                words = list(np.arange(PARALLEL_mdl.d))
                try:
                    start = time.time()
                    PARALLEL_normalized_theta_hat, PARALLEL_theta_hat = PARALLEL_mdl.fit_NewtonRaphson( num_iter= 100)
                    end = time.time()
                    time_took = end - start
                    print("time spent in 1 Bootstrap sample, (DGP model='MNL') : ",time_took)

                except:
                    print("Oops!", sys.exc_info()[0], "occurred.")
                    print("Skipping this seed.")
                    continue
            
            
            b_idx_theta_list.append(PARALLEL_normalized_theta_hat)
            sq_diff.append( (PARALLEL_normalized_theta_hat - normalized_thetaTRUE_mat)**2 )
            abs_diff.append(  np.average(np.abs(PARALLEL_normalized_theta_hat - normalized_thetaTRUE_mat) ))
            _diff.append(  np.average(PARALLEL_normalized_theta_hat - normalized_thetaTRUE_mat))
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
    print("MSE: "  + str(np.asarray(sq_diff,  dtype=np.float32).mean()))
    print("bias: "  + str(np.asarray(_diff,  dtype=np.float32).mean()))
    print("average (over parameters) Variance : "  + str(get_variance_from_b_idx_theta_list(b_idx_theta_list)))
    print("average (over parameters) absolute difference : " + str(np.asarray(abs_diff, dtype=np.float32).mean()))
    print("average time " + str(np.mean(time_list)))