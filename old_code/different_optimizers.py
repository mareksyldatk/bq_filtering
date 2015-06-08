# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 14:30:40 2014

@author: marek
"""

from scipy.optimize import fmin_l_bfgs_b
from DIRECT import solve

def objective(x, args):
    # Parabola in 2 dimensions
    x1 = x[0]
    x2 = x[1]
    f = (x1 - 0.8)**2 + (x2 + 1.2)**2 + 1
    return f

bounds = [(-3, 3), (3, 2)]
x = fmin_l_bfgs_b(objective, x0, bounds)

fmin_cobyla(objective, [0.0, 0.1], [, constr2], rhoend=1e-7)


>>> l = [-3, -2]
>>> u = [3, 2]