# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 21:55:26 2014

@author: marek
"""
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import GPy
from DIRECT import solve
import MatrixOperations as mo

def function(x):
#    y = np.array([np.sin(x[0]), np.cos(x[1]), np.cos(x[0])+np.sin(x[1])])
    y = np.array([np.sin(x[0])])
    return(y)
    
def pi(x, mu, Sigma):
    return(scipy.stats.multivariate_normal.pdf(x, mu, Sigma))

def to_column(x):
    return(x.reshape(-1,1))

def to_row(x):
    return(x.reshape(1,-1))
    
# Prior
x0 = np.array([[0]])
p0 = np.diag([3**2])

# Initial X
X  = np.array([[0],[-1], [1]])
Y  = np.apply_along_axis(function, 1, X)

# Dimensions
X_DIM = X.shape[1]
Y_DIM = Y.shape[1]

# Fit GP
kernel = GPy.kern.RBF(input_dim = X_DIM, ARD=True)
gp = GPy.models.GPRegression(X,Y,kernel)
gp.rbf.variance.constrain_fixed(1.0, warning=False)
gp.Gaussian_noise.variance.constrain_fixed(0.0001**2, warning=False) 
gp.rbf.lengthscale.constrain_fixed(1.0, warning=False)
#gp.rbf.lengthscale.constrain_bounded(0.1,5.0, warning=False)
try:
    print("Optimizing the GP!")
    gp.optimize_restarts(num_restarts=16, verbose=False, parallel=False)
    print("GP optimization done!")
except:
    print("All parameters fixed, GP optimization skipped! :)")    
#gp.plot()

# Optimization bounds
sigma_diag = np.sqrt(np.diag(p0))
lower_const = (x0 - 3*sigma_diag)[0].tolist()
upper_const = (x0 + 3*sigma_diag)[0].tolist()

# Optimization objective
def optimization_objective(X, user_data):
    ''' Optimization objective for DIRECT optimizer:
    - X: point to evaluate,
    - gp: GP object
    - mu, Sigma: prior mean and cov 
    '''
    gp     = user_data['gp']
    mu     = user_data['mu'].squeeze()
    Sigma  = user_data['Sigma']

    X = to_row(X)
    
    _, gp_cov = gp.predict(X)
    pi_x = pi(X, mu, Sigma)
    cost = (pi_x**2 * gp_cov).squeeze()
    return( -cost , 0 )
    
# Find new sample
user_data  = {"gp":gp, "mu":x0, "Sigma":p0}

x_star, _, _ = solve(optimization_objective, lower_const, upper_const, 
                             user_data=user_data, algmethod = 1, 
                             maxT = 3000, maxf = 10000)

x_star    = np.reshape(x_star, (1, X_DIM))                             
y_star, _ = gp.predict(x_star) 
X = np.vstack((X, x_star))
Y = np.vstack((Y, y_star))

#gp = GPy.models.GPRegression(X,Y,kernel)
#gp.optimize_restarts(num_restarts=16, verbose=False, parallel=False)

# Compute integral
cov_Q = np.eye(Y_DIM)
N = len(X)
# Fitted GP parameters      
w_0 = gp.rbf.variance.tolist()[0]
w_d = gp.rbf.lengthscale.tolist()
# Prior parameters
A = np.diag(w_d)
I = np.eye(X_DIM)

def compute_z(a,A,b,B,I,w_0):
    # Make sure of column vectors:
    a, b = to_column(a), to_column(b)
    # Compute z:
    denominator = np.sqrt(np.linalg.det((mo.mldivide(A,B)+I)))
    z = (w_0/denominator) * np.exp(-.5*((a-b).T).dot(mo.mldivide((A+B),(a-b))))   
    return(z[0][0])

# - compute z
z = [compute_z(a, A, x0, p0, I, w_0) for a in X]
z = to_column(np.array(z))

K = gp.kern.K(X)
K = (K.T + K)/2.0

mu = (z.T).dot( mo.mldivide(K, Y) )
W = mo.mrdivide(z.T, K).squeeze().tolist()

_Sigma = None
_CC    = None

for i in range(0,len(z)):
    Y_squared_i = ( to_column(Y[i]-mu) ).dot( to_row(Y[i]-mu) )
    _Sigma       = W[i] * Y_squared_i if i == 0 else _Sigma + W[i] * Y_squared_i
    
    XY_squared_i = ( to_column((X[i]-x0)) ).dot( to_row(Y[i]-mu) )
    _CC           = W[i] * XY_squared_i if i == 0 else _CC + W[i] * XY_squared_i
    
_Sigma = _Sigma + cov_Q 
Sigma = (_Sigma + _Sigma.T)/2.0
CC = _CC