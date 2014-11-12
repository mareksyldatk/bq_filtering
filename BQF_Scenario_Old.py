# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 11:58:28 2014

@author: Marek Syldatk

Implmentation of Bayesian Quadrature Filtering with active sampling

"""
from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as pp
from DIRECT import solve
import GPy
import random
import matrix_operations as mo
import time
import sys


RAND_SEED = 7
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)
    
#%%
# Process model: x_{k+1} = f(x_k)
def model_f(x_k, k=0):
    # return 0.5*x_k + 25*(x_k/(1+(x_k**2))) + 8*np.cos(1.2*k)
    # return x_k 
    return x_k
     
# Measurement model: y_k = h(x_k)
def model_h(x_k):
    # return 0.05*(x_k**2)
    return x_k
    # return 0.05*x_k
    # return 10*np.exp(x_k/10)*np.sin(x_k)

# Prior pdf: N(mu, sigma)
def evaluate_N(x_k, mu, std):
    return sp.stats.norm(mu, std).pdf(x_k)

def appnd(X, X_star):
    ''' Append element to X matrix '''
    X = np.append(X, X_star)
    return(np.array([X]).T)
    
def printl(X):
    print(np.hstack(X))

# Resample weights   
def resample(weights, x):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.rand(), 0
    for u in [(u0+i)/n for i in range(n)]:
        while u > C[j]:
            j+=1
        indices.append(j-1)
    return x[indices] , indices

def normalize(weights):
    w_sum = float(np.sum(weights))
    normalised = [w / w_sum for w in weights]
    return normalised
    
# Cost function to optimize:
def optimization_objective(X, user_data):
    """ Optimization objective for DIRECT """
    gp = user_data['gp']
    m  = user_data['m']
    P  = user_data['P']
    std  = np.sqrt(P)
    ''' FOR DIRECT
    X  = np.array([X])
    _, gp_cov = gp.predict(X)
    pi_x = evaluate_N(X, m, std)
    cost = gp_cov * (pi_x**2)
    return( -cost , 0 )
    '''
    
    ''' For grid search '''
    _, gp_cov = gp.predict(X)
    pi_x = evaluate_N(X, m, std)
    cost = (pi_x**2) * gp_cov
    return(cost)
    
    

# COmpute integral
def compute_z(a,A,b,B,N,I,w_0):
    z = w_0*1/np.sqrt(np.linalg.det((np.linalg.inv(A)).dot(B)+I)) * np.exp(-.5*((a-b).T).dot(np.linalg.inv(A+B)).dot(a-b))  
    return(z)

def optimize_gp(X, Y, gp_kernel):
    gp = GPy.models.GPRegression(X, Y, gp_kernel, normalizer=None)
    #gp.unconstrain('')
    gp.rbf.variance.constrain_fixed(1.0, warning=False)
    gp.Gaussian_noise.variance.constrain_fixed(0.0001**2, warning=False) 
    gp.rbf.lengthscale.constrain_fixed(1.0, warning=False)
    #gp.rbf.lengthscale.constrain_bounded(0.1,5.0, warning=False)
    #gp.optimize_restarts(num_restarts=4, verbose=False, parallel=False)
    return gp
    
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
    
# %%
# ##### ##### #####     BQ INTEGRATION     ##### ##### ##### #
def integrate(N_samples, m, P, cov_Q, fun, **kwargs):
    ''' For a prior of a form N(m, P) and function f(x), together
    with a Normal kernel and observation noise 
    N_samples: number of samples
    m: prior mean
    P: prior covaraincce
    cov_Q: observation noise covaraince
    fun: function
    **kwargs: optiona arguments to fun
    ''' 
    # Parameters:
    _DIM        = 1
    _SIGMA_DIST = 3
    _MAX_T      = 33
    _MAX_F      = 3*_MAX_T
    gp_kernel   = GPy.kern.RBF(input_dim=_DIM)   
    
    # Choose one of two alternatives:
    GRID_STEP = 0.1
    GRID_SIZE = 100
    
    # Initial samples    
    X = np.array([[m]])    
    Y = fun(X, **kwargs)
  
    # Fit GP
    gp = optimize_gp(X, Y, gp_kernel)
    
    for i in range(0, N_samples):
        # Find more samples:        
        u_data  = {"gp":gp, "m":m, "P":P}
        #l_const = [m - _SIGMA_DIST*np.sqrt(P)]
        #u_const = [m + _SIGMA_DIST*np.sqrt(P)]
        ''' hack: expand the search area to cover both modes '''
        l_const = [-np.abs(m) - _SIGMA_DIST*np.sqrt(P)]
        u_const = [ np.abs(m) + _SIGMA_DIST*np.sqrt(P)]
        
        ''' Hack: instead of direct, to a grid search ''' 
        # grid = np.arange(l_const[0],u_const[0]+GRID_STEP,GRID_STEP)
        grid = np.linspace(l_const[0],u_const[0],GRID_SIZE)
        grid = grid.reshape(-1,1)
        objective = optimization_objective(grid, u_data)
        objective = np.hstack(objective).tolist()
        max_ind = objective.index(max(objective))
        x_star = np.array([grid[max_ind]])     
        
        # Solve with direct:
#        x_star, _, _ = solve(optimization_objective, l_const, u_const, 
#                             user_data=u_data, algmethod = 1, 
#                             maxT = _MAX_T, maxf = _MAX_F)        
        # Append sample and refit GP
        X = appnd(X, x_star)
        Y = fun(X, **kwargs)
        gp = optimize_gp(X, Y, gp_kernel)

    ''' hack '''
    # Remove unique rows (and recompute Y):
    X  = unique_rows(X)
    Y  = fun(X, **kwargs)
    gp = optimize_gp(X, Y, gp_kernel)
    ''' hack: end '''
       
    # Number of samples:
    N = len(X)
    # Fitted GP parameters      
    w_0 = gp.rbf.variance
    w_d = gp.rbf.lengthscale
    # Prior parameters
    A = (w_d**2)*np.eye(_DIM)
    I = np.eye(_DIM)

    # Analytical solution 
    ''' TODO: fix it
    K    = gp.kern.K(X)
    K    = (K.T + K)/2.0
    z    = np.array([list(compute_z(a, A, m, P, N, I, w_0)) for a in X])
    E_f  = (z.T).dot( mo.mldivide(K, Y) )
    V_f1 = (w_0)/np.sqrt(  np.linalg.det( 2*mo.mldivide(A, P)  + I )  )
    V_f2 = (z.T).dot( mo.mldivide(K, z) )   
    V_f  = V_f1 - V_f2
    '''
    
    # Compute z's
    z = np.array([list(compute_z(a, A, m, P, N, I, w_0)) for a in X])
    ''' hack '''
    # Instead of matrix inverse use mldivide()
    K = gp.kern.K(X)
    K = (K.T + K)/2.0
    W               = None 
    posterior_mu    = (z.T).dot( mo.mldivide(K, Y) )

    Y_squared       = (Y-posterior_mu) * (Y-posterior_mu)
    posterior_sigma = (z.T).dot( mo.mldivide(K, Y_squared)) + cov_Q 
    
    XY_product      = (X-m) * (Y-posterior_mu)
    posterior_cc    = (z.T).dot( mo.mldivide(K, XY_product) )
    ''' hack: end '''
    
    # Return results:
    return(posterior_mu[0][0], posterior_sigma[0][0], posterior_cc[0][0], 
           {'gp': gp, 'W': W, 'X': X, 'Y': Y, 'z': z, 'K': K,
            'mu':posterior_mu, 'sigma':posterior_sigma, 'cc': posterior_cc,
            'cond': np.linalg.cond(K)})
            #,'E': E_f, 'V': V_f})

# %%
'''
#
#              PROGRAM STARTS HERE 
#
'''
#%%
if __name__ == "__main__":
    pp.close('all')
    
    # ##### ##### #####     SETTINGS     ##### ##### ##### #

    MC_RUNS      = 1      # Number of MC runs
            
    N_TIME_STEPS = 10    # Number of filtering time steps
    N_SAMPLES    = 2      # Number of samples for quadrature filter
    N_PARTICLES  = 1000    # Number of particles for PF
    
    x0_mu          = 0.0
    x0_sigma2      = 1.0
    x_noise_sigma2 = 1.0
    x_noise_std    = np.sqrt(x_noise_sigma2)
    y_noise_sigma2 = 1.0
    y_noise_std    = np.sqrt(y_noise_sigma2)

    Q = x_noise_sigma2
    R = y_noise_sigma2
    
    FILTERING_METHODS = ['KF','UKF', 'PF', 'QF'] # ['QF', 'PF', 'KF', 'UKF']    

    #%% RUN MONTE CARLO
    MC_QF_err = [];  MC_KF_err = [];  MC_PF_err = [];  MC_UKF_err = []
    MC_QF_rmse = []; MC_KF_rmse = []; MC_PF_rmse = []; MC_UKF_rmse = []
    for mc in range(0, MC_RUNS):    
        t = time.time()
        #%% INITIALIZATION:   
        ## BQ
        QF_x_ = []; QF_P_ = []; QF_K_ = []; QF_x__ = []; QF_P__ = []
        QF_ns_p = [None]; QF_ns_u = []
        QF_err = []
        x__ = None; P__ = None ;x_ = None; P_ = None
        ## KF
        F = model_f(1); H = model_h(1)
        KF_x_ = []; KF_P_ = []
        kf_x__ = None; kf_P__ = None; kf_x_ = None; kf_P_ = None
        KF_err = []
        ## PF
        PF_x_  = []; PF_P_  = []
        PF_x__ = []; PF_p__ = []
        pf_x__ = None; pf_P__ = None; pf_x_ = None; pf_P_ = None
        pf_wi  = np.array(normalize([1] * N_PARTICLES))
        PF_err = []
        ## UKF
        UKF_x_ = []; UKF_P_ = []
        ukf_x__ = None; ukf_P__ = None; ukf_x_ = None; ukf_P_ = None
        UKF_err = []
        
        # %%
        # ##### ##### #####     STATES AND OBSERVATIONS     ##### ##### ##### #
        # Generate states 
        x_noise = np.random.normal(0, x_noise_std, N_TIME_STEPS)
        x_true  = np.array([x0_mu])
        for k in range(1,N_TIME_STEPS):
            x_true = np.append(x_true, model_f(x_true[k-1], k) + x_noise[k-1])
            
        # Generate Obeservations
        y_true = np.array([model_h(x) for x in x_true])
        y_noise = np.random.normal(0, y_noise_std, N_TIME_STEPS)
        y = y_true + y_noise
        
        # %%
        # ##### ##### #####     FILTERING     ##### ##### ##### #
        ''' Notation:
            - x__, P__: prediction x_{k|k-1}
            - x_ , P_ : estimate   x_{k}
        '''
        for k in range(0, N_TIME_STEPS):  
            print(str(k) + ' / ' + str(N_TIME_STEPS) + ' / ' + str(mc))
            #%% QUADRATURE FILTER  
            #
            if 'QF' in FILTERING_METHODS:
                #          
                # 1. Prediction
                if(k == 0):
                    old_x, old_P = x0_mu, x0_sigma2
                    x__, P__     = x0_mu, x0_sigma2
                else:
                    old_x, old_P       = x_, P_
                    x__, P__, _, p_par = integrate(N_SAMPLES, x_, P_, Q, model_f, k=k)
                #
                # 2. Update
                mu_, S_, C_, u_par = integrate(N_SAMPLES, x__, P__, R, model_h )    
                #        
                # 3. Get estimates
                K_ = C_/S_
                x_ = x__ + K_ * (y[k] - mu_)
                P_ = P__ - K_ * S_ * K_
                #
                # 4. Save results
                QF_x__ , QF_P__ = appnd(QF_x__, x__) , appnd(QF_P__, P__)
                QF_K_           = appnd(QF_K_,  K_)
                QF_x_ , QF_P_   = appnd(QF_x_,  x_) ,  appnd(QF_P_,  P_)
                if k>0: QF_ns_p = appnd(QF_ns_p, len(p_par['X']))
                QF_ns_u = appnd(QF_ns_u, len(u_par['X']))
            
            #%% KALMAN FILTER
            #
            if 'KF' in FILTERING_METHODS:
                #
                # 1. Prediction
                if(k == 0):
                    kf_x__, kf_P__ = x0_mu, x0_sigma2
                else:
                    kf_x__, kf_P__ = (F * kf_x_) , (F * kf_P_ * F + Q)
                #
                # 2. Update
                kf_S_   = H * kf_P__ * H + R  
                #
                # 3. Get estimates
                kf_K_ = kf_P__ * H / kf_S_   
                kf_x_ = kf_x__ + kf_K_ * (y[k] - H*kf_x__)
                kf_P_ = (1 - kf_K_ * H) * kf_P__
                #
                # 4. Save results
                KF_x_ , KF_P_ = appnd(KF_x_, kf_x_) , appnd(KF_P_, kf_P_)
    
            #%% PARTICLE FILTER   
            #
            if 'PF' in FILTERING_METHODS:
                #     
                # 1. Prediction
                if(k == 0):
                    pf_xi = np.array(np.random.normal(x0_mu, x0_sigma2, N_PARTICLES))
                else:
                    pf_xi, ind = resample(pf_wi, pf_xi)
                    pf_xi = model_f(pf_xi, k=k) + np.random.normal(0, x_noise_std, N_PARTICLES)
                #
                # 2. Update
                pf_wi = evaluate_N(model_h(pf_xi), y[k], y_noise_std)
                pf_wi = normalize(pf_wi)
                #
                # 3. Get estimates
                pf_x_ = np.sum(pf_wi * pf_xi)
                pf_P_ = np.sum(pf_wi * (pf_xi - [pf_x_]*N_PARTICLES)**2 )
                #
                # 4. Save results        
                PF_x_ , PF_P_ = appnd(PF_x_, pf_x_) , appnd(PF_P_, pf_P_)
                
            #%% UNSCENTED KALMAN FILTER   
            #                
            if 'UKF' in FILTERING_METHODS:
                #
                # 0. Precomputations:
                ukf_kappa = 2.0
                ukf_n     = 1.0
                W_i   = np.array([ukf_kappa/(ukf_n + ukf_kappa), 0, 0]) + np.array([0, 1,  1]) * 1/(2*(ukf_n+ukf_kappa))  
                eta_i = np.array([0, 1, -1]) * np.sqrt(ukf_n + ukf_kappa)
                # 1. Prediction
                if(k == 0):
                    ukf_x__, ukf_P__ = x0_mu, x0_sigma2
                else:
                    ukf_Xi__ = ukf_x_ + np.sqrt(ukf_P_)*eta_i
                    ukf_Xi_  = model_f(ukf_Xi__, k=k) 
                    ukf_x__   = np.sum(W_i*ukf_Xi_)
                    ukf_P__   = np.sum(W_i*((ukf_Xi_-ukf_x__)*(ukf_Xi_-ukf_x__))) + Q
                # 2. Update
                ukf_Yi__ = ukf_x__ + np.sqrt(ukf_P__)*eta_i
                ukf_Yi_  = model_h(ukf_Yi__)     
                ukf_mu_   = np.sum(W_i*ukf_Yi_)
                ukf_S_    = np.sum(W_i*((ukf_Yi_-ukf_mu_)*(ukf_Yi_-ukf_mu_))) + R
                ukf_C_    = np.sum(W_i*((ukf_Yi_-ukf_mu_)*(ukf_Yi__-ukf_x__)))
                #        
                # 3. Get estimates
                ukf_K_ = ukf_C_/ukf_S_
                ukf_x_ = ukf_x__ + ukf_K_ * (y[k] - ukf_mu_)
                ukf_P_ = ukf_P__ - ukf_K_ * ukf_S_ * ukf_K_
                #
                # 4. Save results
                UKF_x_ , UKF_P_ = appnd(UKF_x_, ukf_x_) , appnd(UKF_P_, ukf_P_)
                
        ### Printe filtering results:    
        N_ROUND = 4
        print("True X:\n     "), ;print(np.around(x_true, N_ROUND))
        
        if 'QF' in FILTERING_METHODS:
            print("\nBQ Estimates:")
            print("mean:"), ;printl(np.around(QF_x_, N_ROUND))
            print("std: "), ;printl(np.around(np.sqrt(QF_P_), N_ROUND))
            print("#s_p:"), ;printl(QF_ns_p)
            print("#s_u:"), ;printl(QF_ns_u)  
            QF_err  = x_true - np.hstack(QF_x_)
            QF_rmse = np.sqrt(np.sum(QF_err**2)/N_TIME_STEPS) 
            print("err: "), ;print(np.around(QF_err, N_ROUND))
            print("rmse:"), ;print(QF_rmse)
            # Save results of MC run:   
            MC_QF_err  = QF_err if mc==0 else np.vstack((MC_QF_err, QF_err))
            MC_QF_rmse = appnd(MC_QF_rmse, QF_rmse) 
        if 'KF' in FILTERING_METHODS:
            print("\nKF Estimates:") 
            print("mean:"), ;printl(np.around(KF_x_, N_ROUND))
            print("std: "), ;printl(np.around(np.sqrt(KF_P_), N_ROUND))
            KF_err = x_true - np.hstack(KF_x_)
            KF_rmse = np.sqrt(np.sum(KF_err**2)/N_TIME_STEPS) 
            print("err: "), ;print(np.around(KF_err, N_ROUND))
            print("rmse:"), ;print(KF_rmse)
            # Save results of MC run:   
            MC_KF_err  = KF_err  if mc==0 else np.vstack((MC_KF_err, KF_err))
            MC_KF_rmse = appnd(MC_KF_rmse, KF_rmse)
        if 'PF' in FILTERING_METHODS:
            print("\nPF Estimates:") 
            print("mean:"), ;printl(np.around(PF_x_, N_ROUND))
            print("std: "), ;printl(np.around(np.sqrt(PF_P_), N_ROUND))
            PF_err = x_true - np.hstack(PF_x_)
            PF_rmse = np.sqrt(np.sum(PF_err**2)/N_TIME_STEPS) 
            print("err: "), ;print(np.around(PF_err, N_ROUND))
            print("rmse:"), ;print(PF_rmse)
            # Save results of MC run:   
            MC_PF_err  = PF_err  if mc==0 else np.vstack((MC_PF_err, PF_err))
            MC_PF_rmse = appnd(MC_PF_rmse, PF_rmse)
        if 'UKF' in FILTERING_METHODS:
            print("\nUKF Estimates:") 
            print("mean:"), ;printl(np.around(UKF_x_, N_ROUND))
            print("std: "), ;printl(np.around(np.sqrt(UKF_P_), N_ROUND))
            UKF_err = x_true - np.hstack(UKF_x_)
            UKF_rmse = np.sqrt(np.sum(UKF_err**2)/N_TIME_STEPS) 
            print("err: "), ;print(np.around(UKF_err, N_ROUND))
            print("rmse:"), ;print(UKF_rmse)
            # Save results of MC run:   
            MC_UKF_err  = UKF_err  if mc==0 else np.vstack((MC_UKF_err, UKF_err))
            MC_UKF_rmse = appnd(MC_UKF_rmse, UKF_rmse)
        print( time.time() - t )

#%% PRINT MC RESUTS
print("\n\n\n")
if 'QF' in FILTERING_METHODS:
    print("MC: QF rmse mean:"), ;print(np.mean(np.hstack(MC_QF_rmse)))
    print("MC: QF rmse std:"),  ;print(np.std(np.hstack(MC_QF_rmse)))
if 'KF' in FILTERING_METHODS:
    print("MC: KF rmse mean:"), ;print(np.mean(np.hstack(MC_KF_rmse)))
    print("MC: KF rmse std:"),  ;print(np.std(np.hstack(MC_KF_rmse)))
if 'PF' in FILTERING_METHODS:
    print("MC: PF rmse mean:"), ;print(np.mean(np.hstack(MC_PF_rmse)))
    print("MC: PF rmse std:"),  ;print(np.std(np.hstack(MC_PF_rmse)))
if 'UKF' in FILTERING_METHODS:
    print("MC: UKF rmse mean:"), ;print(np.mean(np.hstack(MC_UKF_rmse)))
    print("MC: UKF rmse std:"),  ;print(np.std(np.hstack(MC_UKF_rmse)))

#%% PLOT: Filtering results
pp.rcParams['lines.linewidth'] = 2
fig = pp.figure()  
pp.plot(x_true, '-k')
pp.plot(QF_x_,  '-r')
pp.plot(QF_x_-1.96*np.sqrt(QF_P_),  '--r', linewidth=1)
pp.plot(QF_x_+1.96*np.sqrt(QF_P_),  '--r', linewidth=1)
pp.plot(PF_x_,  '-g')
pp.plot(PF_x_-1.96*np.sqrt(PF_P_),  '--g', linewidth=1)
pp.plot(PF_x_+1.96*np.sqrt(PF_P_),  '--g', linewidth=1)
pp.plot(UKF_x_,  '-b')
pp.plot(UKF_x_-1.96*np.sqrt(UKF_P_),  '--b', linewidth=1)
pp.plot(UKF_x_+1.96*np.sqrt(UKF_P_),  '--b', linewidth=1)
pp.title("Estimates")


#%% PLOT: GP Fitting
fig = pp.figure()
plot_x = np.linspace(-30,30,240)
# Prediction
ax1 = fig.add_subplot(211)
p_par['gp'].plot(ax=ax1)
pp.plot(plot_x, model_f(plot_x, k), '--r')
pp.axvline(old_x, color='r')
pp.axvline(old_x-1.96*np.sqrt(old_P), color='b')
pp.axvline(old_x+1.96*np.sqrt(old_P), color='b')

#ax2 = fig.add_subplot(222)
#pp.plot(plot_x, evaluate_N(plot_x, x_hat, np.sqrt(P_hat)))

# Update
ax3 = fig.add_subplot(212)
u_par['gp'].plot(ax=ax3) 
pp.plot(plot_x, model_h(plot_x), '--r')
pp.axvline(x__, color='r')
pp.axvline(x__-1.96*np.sqrt(P__), color='b')
pp.axvline(x__+1.96*np.sqrt(P__), color='b')

#ax4 = fig.add_subplot(224)
#pp.plot(plot_x, evaluate_N(plot_x, m_k, np.sqrt(P_k)))

#%% 
'''
Some results:
-----
Observations:
- for 3 samples UKF and QF get the same results
- for 10 samples QF beats UKF both in terms of rmse mean and var
- for 25 samples QF beats UKF in a similar way as for 9 samples

Problems:
- BQ picks the wrong mode
- Hack: initiate active sampler with m and -m
- Optimization/Fixed kernel - which should we focus on
- Alternative optimization method to direct (direct seems slow)!
- Analyze the code with Mike
- 

'''