# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 11:11:05 2014

@author: Marek Syldatk

BAYESIAN QUADRATURE WITH ACTIVE SAMPLING FOR FILTERING

SCENARIO 1

LINEAR MOTION MODEL
LINEAR OBSERVATION MODEL
DIM X=4, Y=2

"""
from __future__ import division

import BayesianQuadratureFiltering as bqf
import matplotlib.pyplot as pp
import numpy as np
import random
import GPy

pp.close('all')

# Set random SEED:    
RAND_SEED = 1
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)
    
# Parameters:
N_TIME_STEPS = 25
KAPPA = 2.0
N_PARTICLES = 100
    
# Define model:
model = bqf.SystemModel(f_type='linear', h_type='linear')
T = 1
model.A = np.array([[.5, 0, T, 0], [0, .5, 0, T], [0, 0, T, 0], [0, 0, 0, T]])
model.B = np.array([[T**2/2, 0], [0, T**2/2], [T, 0], [0, T]])
model.F = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
model.G = np.eye(2)
    
model.Q = np.diag([ 1, 1])
model.R = np.diag([ 1, 1]) * .5
    
# Initial conditions:
x0 = np.array([[0, 0, 0, 0]])
p0 = np.diag([1, 1, 1, 1])
    
# Define states and observationsA
X_noise = np.random.multivariate_normal(np.array([0,0]), model.Q, N_TIME_STEPS)
Y_noise = np.random.multivariate_normal(np.array([0,0]), model.R, N_TIME_STEPS)
X_true  = x0
# - States:
for k in range(1,N_TIME_STEPS):
    x_next = model.f(X_true[k-1], k=k) + (model.getB()).dot(X_noise[k-1])
    X_true = np.vstack([X_true, x_next]) 
#  - Obeservations:
Y_true  = np.array([model.h(x) for x in X_true])
Y = Y_true + Y_noise

''' ----- KF ----- '''
kf = bqf.KalmanFilter(model)
kf_X_, X__, P_, P__ = kf.filtering(x0, p0, Y)
        
''' ----- UKF ----- '''
ukf = bqf.UnscentedKalmanFilter(model, kappa=KAPPA)
ukf_X_, X__, P_, P__ = ukf.filtering(x0, p0, Y)   

''' ----- PF ----- '''
pf = bqf.ParticleFilter(model, n_particles=N_PARTICLES)
pf_X_, X__, P_, P__ = pf.filtering(x0, p0, Y)
    
''' ----- QF ----- '''
N_SAMPLES = 32
KERNEL    = GPy.kern.RBF(input_dim = 4, ARD=True)  
    
def K_CONST(gp):
    # Toggle off optimization
    NUM_RESTARTS = 0
    
    gp.rbf.variance.constrain_fixed(1.0, warning=False)        
    gp.rbf.lengthscale.constrain_fixed(3.0, warning=False)
    gp.Gaussian_noise.variance.constrain_fixed(0.0001**2, warning=False) 
     
    return(gp, NUM_RESTARTS)
      
OPT_PAR = {"MAX_T": 333, "MAX_F": 333}
qf = bqf.QuadratureFilter(model, KERNEL, N_SAMPLES, K_CONST, OPT_PAR)
qf_X_, qf_X__, qf_P_, qf_P__ = qf.filtering(x0, p0, Y)
   
# Plot filtering results    
pp.figure()
for i in range(0,4):
    pp.subplot(2,2,i+1)
    pp.plot(X_true[:,i], '-k')
    pp.plot(kf_X_[:,i], '-r')
    pp.plot(ukf_X_[:,i], '--g')
    pp.plot(pf_X_[:,i], '.-b')
    pp.plot(qf_X_[:,i], '.-m')
    
# Print RMSE:
print ("RMSE for EKF: " + str( np.sqrt(np.sum(np.power((X_true- kf_X_),2))/N_TIME_STEPS) ) )
print ("RMSE for UKF: " + str( np.sqrt(np.sum(np.power((X_true-ukf_X_),2))/N_TIME_STEPS) ) )
print ("RMSE for PF:  " + str( np.sqrt(np.sum(np.power((X_true- pf_X_),2))/N_TIME_STEPS) ) )
print ("RMSE for QF:  " + str( np.sqrt(np.sum(np.power((X_true- qf_X_),2))/N_TIME_STEPS) ) )