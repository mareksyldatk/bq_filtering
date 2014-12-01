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

#%% sample inputs and outputs
N = 10
#X = np.random.uniform(-3.,3.,(N,2))
#Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(N,1)*0.05

X = to_column(np.linspace(0,np.pi,N))
Y = np.sin(X) + np.random.randn(N,1)*0.05


#%% define kernel
ker = GPy.kern.RBF(1,ARD=True)

#%% create simple GP model
gp = GPy.models.GPRegression(X,Y,ker)
gp.rbf.variance.constrain_positive(warning=False)
gp.rbf.lengthscale.constrain_fixed(3.0 ,warning=False)
gp.rbf.variance.constrain_fixed(1.0, warning=False)        
gp.Gaussian_noise.variance.constrain_fixed(0.001**2, warning=False)

#def dK_theta(gp, X1, X2, Y, theta, ARD=True):
#    ''' Derivative of K with respect to \theta '''
#    dK_theta = None
#    return(dK_theta)
   
#%%
   
ARD = True
theta = [1.0, 3.0]    
X_star = to_row(X[0]) + np.random.rand(1,2)
X_data = X
Y      = Y
input_dim = X.shape[1]
theta_0 = np.array(gp.kern.variance.tolist())
theta_i = np.array(gp.kern.lengthscale.tolist())

K_x  = gp.kern.K(X_star)
K_xd = gp.kern.K(to_row(X_star), X_data)
K    = gp.kern.K(X_data)

# Compute d_m/d_theta_0
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

if (ARD == False):
    dKx_thetai  = np.sum(dKx_thetai,0)
    dKxd_thetai = np.sum(dKxd_thetai,0)
    dK_thetai   = np.sum(dK_thetai,0)[np.newaxis]

dKx_theta  = np.vstack((dKx_theta0,  dKx_thetai))
dKxd_theta = np.vstack((dKxd_theta0, dKxd_thetai))
dK_theta   = np.vstack((dK_theta0,   dK_thetai))

#%%
D = dKxd_theta.shape[0]
dm_theta = np.zeros((D,1))
dV_theta = np.zeros((D,1))

for i in range(0, input_dim):
    dm_theta[i] = (  mrdivide(to_row(dKxd_theta[i]), K)  -  K_xd.dot( mldivide(K, mrdivide(dK_theta[i], K)) )  ).dot(Y)
    dV_theta[i] = ( 
        dKx_theta[i]
        - (dKxd_theta[i]).dot(mldivide(K, to_column(K_xd)))
        - (dKxd_theta[i]).dot( mldivide(dK_theta[i], mrdivide(dK_theta[i],dK_theta[i])) ).dot(to_column(dKxd_theta[i]))
        + (K_xd).dot(mldivide(K, to_column(dKxd_theta[i]) ))
        )
        
#%% 
Sigma = (0.001**2) * np.eye(D)
m, V = gp.predict(X_star)
m_tilde = m
V_tilde = (4.0/3.0) * V + (dm_theta.T).dot(Sigma.dot(dm_theta)) + (1.0/3.0*V)*(dV_theta.T).dot(Sigma.dot(dV_theta))
print(V, V_tilde)
