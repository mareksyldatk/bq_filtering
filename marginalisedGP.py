# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 14:27:09 2014

@author: marek
"""
from __future__ import division
import numpy as np
import GPy
import matplotlib.pyplot as plt

def to_column(x):
    if (x.__class__ == np.ndarray):
        return(x.reshape(-1,1))
    elif (x.__class__ == list):
        return(np.array([x]).reshape(-1,1))
    else:
        return(np.array([[x]]).reshape(-1,1))

def to_row(x):
    if (x.__class__ == np.ndarray):
        return(x.reshape(1,-1))
    elif (x.__class__ == list):
        return(np.array([x]).reshape(1,-1))
    else:
        return(np.array([[x]]).reshape(1,-1))
        
def mldivide(A,B):
    ''' Solves A x = B ==> x = A^{-1} B
        Corresponds to Matlabs: A/B 
    '''
    x_solve = np.linalg.solve(A,B)
    # x_lstsq = np.linalg.lstsq(A,B)
    return(x_solve)

def mrdivide(B,A):   
    ''' Solves x A = B ==> x = B A^{-1}
        x = ( (A^T)^{-1} B^T )^T 
        Corresponds to matlabs B/A
    '''
    x_solve = (np.linalg.solve(A.T,B.T)).T
    # x_lstsq = (np.linalg.lstsq(A.T,B.T)).T
    return(x_solve)
 
def mGP_predict(X_star, gp, Sigma):    
    ARD = gp.kern.ARD
    X_star = to_row(X_star)    
    X_data = gp.X
    Y      = gp.Y
    input_dim = X_data.shape[1]
    
    # Form theta vector
    theta_0 = np.array(gp.kern.variance.tolist())
    theta_i = 1/np.array(gp.kern.lengthscale.tolist())
    theta   = np.vstack((theta_0, theta_i))
    
    # Precompute kernels
    K_x  = gp.kern.K(X_star)
    K_xd = gp.kern.K(to_row(X_star), X_data)
    K    = gp.kern.K(X_data)

    # Compute derivatives of kernels
    dKx_theta0  = ( K_x/theta_0 )
    dKxd_theta0 = ( K_xd/theta_0 )
    dK_theta0   = ( K/theta_0 )[np.newaxis]
    
    dKx_thetai = dKxd_thetai = dK_thetai = None
    for i in range(0, input_dim):
        _dKx_thetai   = 0.0
        _dKxd_thetai = - K_xd * ( X_data[:,i] - X_star[:,i] )**2
        _dK_thetai   = - K * ( to_column(X_data[:,i]) - to_row(X_data[:,i]) )**2
        
        dKx_thetai  = _dKx_thetai  if i == 0 else np.vstack((dKx_thetai,  _dKx_thetai))
        dKxd_thetai = _dKxd_thetai if i == 0 else np.vstack((dKxd_thetai, _dKxd_thetai))
        dK_thetai   = _dK_thetai[np.newaxis]  if i == 0 else np.vstack((dK_thetai, _dK_thetai[np.newaxis]))
    
    # For ard sum 
    if (ARD == False):
        dKx_thetai  = np.sum(dKx_thetai,0)
        dKxd_thetai = np.sum(dKxd_thetai,0)
        dK_thetai   = np.sum(dK_thetai,0)[np.newaxis]
    
    dKx_theta  = np.vstack((dKx_theta0,  dKx_thetai))
    dKxd_theta = np.vstack((dKxd_theta0, dKxd_thetai))
    dK_theta   = np.vstack((dK_theta0,   dK_thetai))
    
    # Comute dm and dV
    D = len(theta)
    dm_theta = np.zeros((D,1))
    dV_theta = np.zeros((D,1))
    
    for i in range(0, D):
        dm_theta[i] = (  mrdivide(to_row(dKxd_theta[i]), K)  -  K_xd.dot( mldivide(K, mrdivide(dK_theta[i], K)) )  ).dot(Y)
        dV_theta[i] = ( 
            dKx_theta[i]
            - (dKxd_theta[i]).dot(mldivide(K, to_column(K_xd)))
            - (dKxd_theta[i]).dot( mldivide(dK_theta[i], mrdivide(dK_theta[i],dK_theta[i])) ).dot(to_column(dKxd_theta[i]))
            + (K_xd).dot(mldivide(K, to_column(dKxd_theta[i]) ))
            )
            
    # Return predicted m, V, m_tilde and v_tilde
    m, V = gp.predict(X_star)
    V_tilde = (4.0/3.0) * V + (dm_theta.T).dot(Sigma.dot(dm_theta)) + (1.0/3.0*V)*(dV_theta.T).dot(Sigma.dot(dV_theta))
    return(m, V, V_tilde)

def mGP_apply(X_star, gp, Sigma):
    result = np.apply_along_axis( mGP_predict, 1, X_star, gp, Sigma  )
    m = result[:,0]
    V = result[:,1]
    V_tilde = result[:,2]   
    cost_bald = to_column(V_tilde / V) 
    cost_uncertainty = to_column(V_tilde)
    return( cost_bald, cost_uncertainty, m, V, V_tilde)

''' 
            SIMULATION   
'''

# %% MODEL
def model_f(x):
    par = {'delta': 0.5, 'mu': 0.0, 'sigma': 0.75, 'A': 0.5, 'phi': 10.0, 'offset': 0.0}    
    # Compute components of y:
    y_a = (1.0 - np.exp(par['delta']*x)/(1+np.exp(par['delta']*x)))
    y_b = np.exp(-(x-par['mu'])**2 / par['sigma'])
    y_c = par['A']* np.cos(par['phi']*x)
    # Return y:
    y = y_a + y_b * y_c + par['offset']
    return(y)

# Generate intial data:   
N = 20
X = to_column(-2*np.pi * np.random.rand(N) + np.pi)
Y = model_f(X)

# Kernel and hyperparameters
ker = GPy.kern.RBF(1,ARD=True)
theta = [0.35, 0.26]
theta_Sigma = np.diag([1, 1])**2
gp = GPy.models.GPRegression(X,Y,ker)
gp.rbf.variance.constrain_positive(warning=False)
gp.rbf.variance.constrain_fixed(theta[0], warning=False)
gp.rbf.lengthscale.constrain_fixed(theta[1] ,warning=False)
gp.Gaussian_noise.variance.constrain_fixed(0.01**2, warning=False)
# gp.optimize_restarts(num_restarts=10)

# Sample interval:
X_star = to_column(np.linspace(-np.pi, np.pi, 1000))
Y_star = model_f(X_star)
cost_bald, cost_uncertainty, m, V, V_tilde = mGP_apply(X_star, gp, theta_Sigma)

# Plot
plt.close('all')
fig = plt.figure()

fig.add_subplot(211)
plt.plot(X_star, Y_star, '-b')
plt.plot(X_star, m, '-r')
plt.plot(X_star, m-np.sqrt(V), '--r'); plt.plot(X_star, m+np.sqrt(V), '--r')
plt.plot(X_star, m-np.sqrt(V_tilde), '--g'); plt.plot(X_star, m+np.sqrt(V_tilde), '--g')

fig.add_subplot(212)
plt.plot(X_star, cost_bald )
plt.plot(X_star, cost_uncertainty,'-r' )