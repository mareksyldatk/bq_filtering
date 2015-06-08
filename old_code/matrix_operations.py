# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 10:46:53 2014

@author: marek
"""

import numpy as np

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
    

if __name__ == "__main__":
    ''' Example 1 '''
    A = np.array([[1, 2 ,3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[15], [15], [15]])    
    # Matlab: x = [-39, 63, -24]
    print(mldivide(A,B))
    
    C = np.array([[1,1,1],[1,1,1],[1,1,1]])
    mldivide(C,B)
    
    ''' Example 2 ''' 
    A = np.array([[1, 1 ,3], [2, 0, 4], [-1, 6, -1]])
    B = np.array([[2, 19, 8]])
    # Matlab: x = [1.0000 2.0000 3.0000]
    print(mrdivide(B,A))

    
    
