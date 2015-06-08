# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 15:37:46 2014

@author: Marek Syldatk

Using additive kenrel for fitting GP
"""

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as pp
from DIRECT import solve
import GPy
import random
import sys
import collections

import matrix_operations as mo

pp.close('all')

#%%
# Process model: x_{k+1} = f(x_k)
def function(x_k, k=0):
    # return 0.5*x_k + 25*(x_k/(1+x_k**2)) + 8*np.cos(1.2*(k+1))
    return 3*x_k**2 + 2
    # return 2*x_k
    
# Prior pdf: N(mu, sigma)
def evaluate_prior(x_k, mu, sigma):
    return sp.stats.norm(mu, sigma).pdf(x_k)

#%% Determine prior
mu        = 100000
sigma2    = 1**2
sigma     = np.sqrt(sigma2)
N_samples = 5


#%% Fit GP
X = np.linspace(-3*sigma+mu, +3*sigma+mu, N_samples)
X = np.array([[x] for x in X])
Y = function(X)

kern_rbf  = GPy.kern.RBF(input_dim=1)
kern_lin  = GPy.kern.Linear(input_dim=1)
#kern_lin  = GPy.kern.LinearFull(input_dim=1, rank=1)
kern_bias = GPy.kern.Bias(input_dim=1)
kernel    =  kern_rbf + kern_lin

meanY = np.mean(Y)
meanX = np.mean(X)
Y = Y - meanY
X = X - meanX

gp = GPy.models.GPRegression(X, Y, kernel, normalizer=None)
gp.Gaussian_noise.variance.constrain_fixed(0.0001**2)
gp.optimize_restarts(64, verbose=False)

newX = np.arange(-3*sigma+mu, +3*sigma+mu, 0.1) 
newX = newX.reshape(-1,1)
newY, covY = gp.predict(newX-meanX)
newY = newY + meanY

pp.plot(newX,function(newX),'-g')
pp.plot(newX,newY,'--r')

print(gp)
