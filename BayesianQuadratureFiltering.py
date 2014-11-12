# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 17:22:35 2014

@author: marek
"""
from __future__ import division
from DIRECT import solve
import MatrixOperations as mo
import numpy as np
import scipy.stats
import GPy

'''
Nonation:
    - Filtering:
            - x_  = x_{k|k}
            - x__ = x{k|k-1}
        - Smoothing    
            - _x  = x{k|K}
'''    
# HELP FUNCTIONS:
def symetrize_cov(P):
    return (P + P.T)/2.0
    
def to_column(x):
    return(x.reshape(-1,1))

def to_row(x):
    return(x.reshape(1,-1))
    
def unique_rows(x):
    x = np.ascontiguousarray(x)
    unique_x = np.unique(x.view([('', x.dtype)]*x.shape[1]))
    return unique_x.view(x.dtype).reshape((unique_x.shape[0], x.shape[1]))

# MODEL
class SystemModel(object):
    ''' Model Object for a model described by the following model equations in:
        - linear case:
            x__ = Ax_ + Bv
            y_  = Hx_ + Gw
        - nonlinear case:
            x__ = f(x_) + Bv
            y_  = h(x_) + Gw
    '''
    
    # Constructor:
    def __init__(self, f_type='linear', h_type='linear'):
        # Define type of model:
        self._f_type = f_type
        self._h_type = h_type
        
        # Linear model parameters:
        self.A = None
        self.B = None
        self.F = None
        self.G = None
    
        # Nonlinear model functions:
        self.fun_f = None
        self.fun_h = None
        
        # Noise parameters:
        self.Q = None
        self.R = None
        
        # Derivatives of nonlinear model:
        self.der_f = None
        self.der_h = None
        
    # Propagation:   
    def f(self, x_, k=None):
        if self._f_type == 'linear':
            x__ = (self.A).dot(x_) 
        else:
            x__ = self.fun_f(x_, k)
        return(x__)

    # Observation:            
    def h(self, x_, k=None):
        if self._h_type == 'linear':
            y_ = (self.F).dot(x_) 
        else:
            y_ = self.fun_h(x_, k)
        return(y_)
    
    # Return state matrices or linearizations:    
    def getA(self, x_=None, k=None):
        if self._f_type == 'linear':
            return(self.A)
        else:
            return(self.der_f(x_, k))
            
    def getB(self, x_=None, k=None):
        return(self.B)

    def getF(self, x_=None, k=None):
        if self._h_type == 'linear':
            return(self.F)
        else:
            return(self.der_h(x_, k))   
            
    def getG(self, x_=None, k=None):
        return(self.G)

            
#%%
#
# Kalman Filter
#
#
class KalmanFilter(object):
    ''' Kalman FIlter implementation '''
    # Constructor:
    def __init__(self, model):
        # Set model
        self.model = model
        
    def filtering(self, x0, p0, Y):
        # 0. Initiation
        N_TIME_STEPS = len(Y)
        self.X_DIM   = x0.shape[1]
        self.Y_DIM   = Y.shape[1]
        
        x_ = p_ = x__ = p__ = None
        X_ = P_ = X__ = P__ = None
        
        #
        # FILTERING LOOP
        for k in range(0, N_TIME_STEPS): 
            #
            # 1. Prediction
            if(k == 0):
                x__, p__ = x0.T, p0
            else:
                A = self.model.getA(x_, k)
                B = self.model.getB(x_, k)
                Q = self.model.Q
                
                x__, p__ = self.model.f(x_, k) , A.dot(p_).dot(A.T) + B.dot(Q).dot(B.T)
                p__ = symetrize_cov(p__)
            #
            # 2. Update
            F = self.model.getF(x__, k)
            G = self.model.getG(x__, k)
            R = self.model.R
            
            S = F.dot(p__).dot(F.T) + G.dot(R).dot(G.T)  
            #
            # 3. Get estimates
            # K = (p__).dot(mo.mrdivide(F.T, S))
            K = (p__).dot(F.T).dot(np.linalg.inv(S))
            
            y_hat = self.model.h(x__, k)
            eps = np.array([Y[k]]).T - y_hat            
            x_ = x__ + K.dot(eps)
            p_ = (np.eye(self.X_DIM) - K.dot(F)).dot(p__)
            p_ = symetrize_cov(p_)
            
            #
            # 4. Save results
            X__ = x__.T           if k==0 else np.vstack([X__, x__.T])
            P__ = np.array([p__]) if k==0 else np.vstack([P__, [p__]])
            X_  = x_.T            if k==0 else np.vstack([X_,  x_.T])   
            P_  = np.array([p_])  if k==0 else np.vstack([P_,  [p_]])
         
        
        return(X_, X__, P_, P__)
        
    def smoothing(self, X_, X__, P_, P__, Y):
        _X = None
        _P = None
        return(_X, _P)

#%%
#
# Unscented Kalman Filter
#
#
class UnscentedKalmanFilter(object):
    # Constructor:
    def __init__(self, model, kappa):
        # Set model
        self.model = model
        self.kappa = kappa
        
    def filtering(self, x0, p0, Y):
        # 0. Initiation
        N_TIME_STEPS = len(Y)
        self.X_DIM   = x0.shape[1]
        self.Y_DIM   = Y.shape[1]
        
        x_ = p_ = x__ = p__ = None
        X_ = P_ = X__ = P__ = None
        
        #
        # FILTERING LOOP
        for k in range(0, N_TIME_STEPS): 
            #
            # 1. Prediction    
            if(k == 0):
                x__, p__ = x0.T, p0
            else:
                B = self.model.getB(x_, k)
                Q = self.model.Q
                
                x__, p__, _ = self.unscented_transform(self.model.f, x_, p_, self.kappa, k=k)
                p__         = p__ + B.dot(Q).dot(B.T)
                p__         = symetrize_cov(p__)
                
            #
            # 2. Update
            G = self.model.getG(x__, k)
            R = self.model.R
            
            mu, S, C = self.unscented_transform(self.model.h, x__, p__, self.kappa, k=k)
            S        = S + G.dot(R).dot(G.T)
            S        = symetrize_cov(S)
            
            #
            # 3. Get estimates
            K = mo.mrdivide(C, S)
            # K = C.dot(np.linalg.inv(S))
            eps = np.array([Y[k]]).T - mu   
            x_ = x__ + K.dot(eps)
            p_ = p__ - K.dot(S).dot(K.T)
            p_ = symetrize_cov(p_)
            
            #
            # 4. Save results
            X__ = x__.T           if k==0 else np.vstack([X__, x__.T])
            P__ = np.array([p__]) if k==0 else np.vstack([P__, [p__]])
            X_  = x_.T            if k==0 else np.vstack([X_,  x_.T])   
            P_  = np.array([p_])  if k==0 else np.vstack([P_,  [p_]])
            
        return(X_, X__, P_, P__)
        
    def smoothing(self, X_, X__, P_, P__, Y):
        _X = None
        _P = None
        return(_X, _P)
        
    def unscented_transform(self, fun, m, P, kappa, **kwargs):
        n      = m.shape[0]
        sqrt_P = np.linalg.cholesky(P)
        
        w_0     = kappa/(n + kappa)
        w_i     = 1.0/(2.0*(n + kappa))
        Wi      = np.append( w_0, np.ones((1,2*n)) * w_i)
        
        xi_0 = np.zeros((1,n))
        xi_n = np.vstack((np.diag([1.0]*n), np.diag([-1.0]*n)))
        Xi   = np.vstack((xi_0, xi_n)) * np.sqrt(kappa+n)
        
        mu = S = C = None
                
        for i in range(0,2*n+1):
            sigma_i = m + sqrt_P.dot(np.array([Xi[i]]).T)
            chi_i   = fun(sigma_i, **kwargs)
            mu_i    = Wi[i]*( chi_i )
            
            mu = mu_i if i==0 else mu + mu_i
            
        for i in range(0,2*n+1):
            sigma_i = m + sqrt_P.dot(np.array([Xi[i]]).T)
            chi_i   = fun(sigma_i, **kwargs)
            S_i     = Wi[i]*( (chi_i - mu).dot((chi_i - mu).T) )
            C_i     = Wi[i]*( (sigma_i - m).dot((chi_i - mu).T)  )
            
            S  = S_i  if i==0 else S + S_i
            C  = C_i  if i==0 else C + C_i
            
        S = (S + S.T)/2.0
        
        return(mu, S, C)
        
#%%
#
# Particle Filter
#
#
class ParticleFilter(object):
    # Constructor:
    def __init__(self, model, n_particles):
        # Set model
        self.model = model
        self.N_PARTICLES = n_particles
        
    def filtering(self, x0, p0, Y):
        # 0. Initiation
        N_TIME_STEPS = len(Y)
        self.X_DIM   = x0.shape[1]
        self.Y_DIM   = Y.shape[1]
        
        x_ = p_ = x__ = p__ = None
        X_ = P_ = X__ = P__ = None
        
        wi = np.array([1]*self.N_PARTICLES)
        wi = self.normalize(wi)
            
        #
        # FILTERING LOOP
        for k in range(0, N_TIME_STEPS): 
            
            #     
            # 1. Prediction
            if(k == 0):
                xi = np.random.multivariate_normal(x0[0], p0, self.N_PARTICLES)
            else:
                B = self.model.getB(x_, k=k)
                Q = self.model.Q
                xi, _ = self.resample(wi, xi)
                #noise = np.random.multivariate_normal([0]*self.X_DIM, B.dot(Q).dot(B.T), self.N_PARTICLES)
                #xi = np.apply_along_axis(self.model.f, 1, xi, k=k) + noise
                
                noise = np.random.multivariate_normal([0]*Q.shape[0], Q, self.N_PARTICLES)
                xi = np.apply_along_axis(self.model.f, 1, xi, k=k) + noise.dot(B.T)
                
            
            # Get predictions
            x__ = np.array(wi).dot(xi)
            x__ = np.array([x__]).T
            
            for i in range(0,self.N_PARTICLES):
                pi__ = wi[i] * (  ((xi[i]-x__.T).T).dot(xi[i]-x__.T)  )
                p__  = pi__ if i==0 else p__ + pi__    
            p__ = symetrize_cov(p__)
            #
            # 2. Update
            G = self.model.getG()
            R = self.model.R
            yi = np.apply_along_axis(self.model.h, 1, xi, k=k)
            wi = np.apply_along_axis(scipy.stats.multivariate_normal(Y[k], G.dot(R).dot(G.T)).pdf, 1, yi)
            wi = self.normalize(wi)
            
            #
            # 3. Get estimates
            x_ = np.array(wi).dot(xi)
            x_ = np.array([x_]).T
            
            for i in range(0,self.N_PARTICLES):
                pi_ = wi[i] * (  ((xi[i]-x_.T).T).dot(xi[i]-x_.T)  )
                p_  = pi_ if i==0 else p_ + pi_
            p__= symetrize_cov(p_)
            #
            # 4. Save results
            X__ = x__.T           if k==0 else np.vstack([X__, x__.T])
            P__ = np.array([p__]) if k==0 else np.vstack([P__, [p__]])
            X_  = x_.T            if k==0 else np.vstack([X_,  x_.T])   
            P_  = np.array([p_])  if k==0 else np.vstack([P_,  [p_]])
            
        return(X_, X__, P_, P__)
        
    def smoothing(self, X_, X__, P_, P__, Y):
        _X = None
        _P = None
        return(_X, _P)
        
    # Resample weights   
    def resample(self, weights, x):
        n = len(weights)
        indices = []
        C = [0.] + [sum(weights[:i+1]) for i in range(n)]
        u0, j = np.random.rand(), 0
        for u in [(u0+i)/n for i in range(n)]:
            while u > C[j]:
                j+=1
            indices.append(j-1)
        return x[indices,:] , indices
            
    def normalize(self, weights):
        w_sum = float(np.sum(weights))
        normalised = [w / w_sum for w in weights]
        return normalised


#%%
#
# Unscented Kalman Filter
#
#
class QuadratureFilter(object):
    # Constructor:
    def __init__(self, model, kern, n_samples=10, k_const=None, opt_par=None):
        # Set model
        self.model     = model
        self.kern      = kern
        self.k_const   = k_const
        self.N_SAMPLES = n_samples
        if (opt_par == None):
            self.opt_par = {"MAX_T": 1000, "MAX_F": 3000}
        else:
            self.opt_par = opt_par
            
            
    def filtering(self, x0, p0, Y):
        # 0. Initiation
        N_TIME_STEPS = len(Y)
        self.X_DIM   = x0.shape[1]
        self.Y_DIM   = Y.shape[1]
        
        x_ = p_ = x__ = p__ = None
        X_ = P_ = X__ = P__ = None
        
        #
        # FILTERING LOOP
        for k in range(0, N_TIME_STEPS): 
            #
            # 1. Prediction    
            if(k == 0):
                x__, p__ = x0.T, p0
            else:
                B = self.model.getB(x_, k)
                Q = self.model.Q
                
                x__, p__, _ = self.integrate(x_, p_, self.model.f, k=k)
                p__         = p__ + B.dot(Q).dot(B.T)
                p__         = symetrize_cov(p__)
            #
            # 2. Update
            G = self.model.getG(x__, k)
            R = self.model.R
            
            mu, S, C = self.integrate(x__, p__, self.model.h, k=k)
            S        = S + G.dot(R).dot(G.T)
            S        = symetrize_cov(S)
            
            #
            # 3. Get estimates
            eps = Y[k] - to_row(mu)
            eps = to_column(eps)
            
            # TODO: Version with K computed
            # K = mo.mrdivide(C, S)
            # K = C.dot(np.linalg.inv(S))
            # x_ = x__ + K.dot(eps)
            # p_ = p__ - K.dot(S).dot(K.T)
            # p_ = symetrize_cov(p_)

            # TODO: Version without direct computation of K
            x_ = x__ + C.dot(mo.mldivide(S, eps))
            p_ = p__ - C.dot(mo.mrdivide(C,S).T)
            p_ = symetrize_cov(p_)       
            
            #
            # 4. Save results
            X__ = x__.T           if k==0 else np.vstack([X__, x__.T])
            P__ = np.array([p__]) if k==0 else np.vstack([P__, [p__]])
            X_  = x_.T            if k==0 else np.vstack([X_,  x_.T])   
            P_  = np.array([p_])  if k==0 else np.vstack([P_,  [p_]])
            
        return(X_, X__, P_, P__)
                
            
    def pi(self, X, m, P):
        return(scipy.stats.multivariate_normal.pdf(X, m, P))
        
    def optimization_objective(self, X, user_data):
        gp = user_data['gp']
        m  = user_data['m']
        P  = user_data['P']
                
        m = to_row(m) 
        X = to_row(X)
             
        _, gp_cov = gp.predict(X)
        pi_x = self.pi(X.squeeze(), m.squeeze(), P)
        cost = (pi_x**2 * gp_cov).squeeze()
        
        return( -cost , 0 )

    def gp_fit(self, X, Y):
        # Regression
        gp = GPy.models.GPRegression(X, Y, self.kern)
        # Constraints
        gp, NUM_RESTARTS = self.k_const(gp) if (self.k_const != None) else (gp, 8)
        # Optimize
        try:
            gp.optimize_restarts(num_restarts=NUM_RESTARTS, verbose=False, parallel=True)
            print("GP optimization finished!") 
        except:
            print("All parameters fixed (or error): GP optimization skipped!") 
        
        return(gp)
        
    def compute_z(self, a, A, b, B, I, w_0):
        ''' Input: 
            - a, b - row vectors 
        '''
        # Make sure of column vectors:
        a, b = to_column(a), to_column(b)
        # Compute z:
        denominator = np.sqrt(np.linalg.det((mo.mldivide(A,B)+I)))
        z = (w_0/denominator) * np.exp(-.5*((a-b).T).dot(mo.mldivide((A+B),(a-b))))   
        return(z[0][0])
        
    def integrate(self, m, P, fun, **kwargs):
        ''' Input:
            m - column vector
            P - matrix
            Output:
            x - column vector
            Variables:
            X, Y - matrix of row vectors
            z - column vector
            m, x, mu - row vector
        ''' 
        
        # Initial sample and fitted GP:
        m = to_row(m)
        X = m
        Y = np.apply_along_axis(fun, 1, X, **kwargs)
        gp = self.gp_fit(X,Y)
        
        # Optimization constraints:
        N_SIGMA = 3
        P_diag = np.sqrt(np.diag(P))
        lower_const = (m - N_SIGMA*P_diag)[0].tolist()
        upper_const = (m + N_SIGMA*P_diag)[0].tolist()
        
        # Perform sampling
        for i in range(0, self.N_SAMPLES):
            # Set extra params to pass to optimizer
            user_data  = {"gp":gp, "m":m, "P":P}
            x_star, _, _ = solve(self.optimization_objective, lower_const, upper_const, 
                             user_data=user_data, algmethod = 1, 
                             maxT = self.opt_par["MAX_T"], 
                             maxf = self.opt_par["MAX_F"])

            x_star = to_row(x_star)                           
            X      = np.vstack((X, x_star))
            Y      = np.apply_along_axis(fun, 1, X, **kwargs)
            gp     = self.gp_fit(X, Y)
            
        # Reoptimize GP:                             
        # TODO: Remove unique rows:
        X  = unique_rows(X)
        Y  = np.apply_along_axis(fun, 1, X, **kwargs)
        gp = self.gp_fit(X, Y)   
            
        # Compute integral
        # Fitted GP parameters      
        w_0 = gp.rbf.variance.tolist()[0]
        w_d = np.power(gp.rbf.lengthscale.tolist(), 2)
        
        # Prior parameters
        A = np.diag(w_d)
        I = np.eye(self.X_DIM)     
        
        # Compute weigths
        z = [self.compute_z(x, A, m, P, I, w_0) for x in X]
        z = to_column(np.array(z))
        K = gp.kern.K(X); K = (K.T + K)/2.0
        W = (mo.mrdivide(z.T, K).squeeze()).tolist()
        
        # Compute mean, covariance and cross-cov
        mu_ = (z.T).dot( mo.mldivide(K, Y) )
        mu_ = to_row(mu_)
        
        # Initiale Sigma and CC
        Sigma_ = CC_ = None     
        
        # TODO: Sigma computation as closed form
        Sigma_ = w_0/np.sqrt(np.linalg.det( 2*mo.mldivide(A,P) + I) ) - (z.T).dot(mo.mldivide(K,z))
        
        # Compute cov matrix and cross cov matrix
        for i in range(0,len(W)):
        
            # TODO: Sigma computation as sumation
            # YY_i   = ( to_column(Y[i]-mu_) ).dot( to_row(Y[i]-mu_) )
            # Sigma_ = W[i] * YY_i if i == 0 else Sigma_ + W[i] * YY_i
            
            XY_i = ( to_column(X[i]-m) ).dot( to_row(Y[i]-mu_) )
            CC_  = W[i] * XY_i if i == 0 else CC_ + W[i] * XY_i
        
        mu_    = to_column(mu_)
        Sigma_ = symetrize_cov(Sigma_)
        
        # Return results
        return(mu_, Sigma_, CC_)
        

