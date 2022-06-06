#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 00:14:42 2022

@author: hazan
"""

import numpy as np
from numpy import linalg as LA
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt

def lmafit(mat, mask, rank, x0 = None, lambda_x=0, lambda_y=0, maxit=2000, kappa = 1-1e-6, tol_res_error = 1e-8):
    '''
        Solves:
            \min_{X,Y} 0.5*(\| P_\Omega(XY) - b \|_2^2 +
                            \lambda_x \|X\|_F^2 + \lambda_y \|Y\|_F^2)

        lambda_x, lambda_y (float): Regularisation parameter, not clear how it should be picked

    '''
    ### MODIFICATION BY BASTIEN ###
    mask_inv = np.ones(mask.shape, dtype = np.int)-mask
    # mask = 1 = True when known and mask = 0 = False when unknown
    # mask_inv = 1 = True when unknown and mask = 0 = False when known
    
    
    data = mat*mask
    
    m, n = mat.shape
    if x0 is None:
        X = np.zeros((m, rank))
        Y = np.eye(rank, n)
    else:
        X = x0[0]
        Y = x0[1]

    Res = data

    Z = np.zeros((m,n))  
    Z += mask*data
    
    Z += mask_inv*(np.sum(data)/np.sum(mask)) # mean of known values
    eye = np.eye(rank)
    b_norm = LA.norm(Res,2)

    rel_error = np.zeros((maxit+1,1))
    rel_error_kappa = -1
    it = 0
    for i in range(maxit):
        it += 1
        
        Xt = np.linalg.solve(np.dot(Y,Y.T) + lambda_x*eye, np.dot(Y,Z.T))
        X = Xt.T
        Y = np.linalg.solve(np.dot(Xt,X) + lambda_y*eye,np.dot(Xt,Z))
        Z = np.dot(X,Y)

        # The correct way to measure error
        rel_error[i] = 0.5*( (LA.norm(Z*mask-Res,2)**2
                    + (lambda_x) *LA.norm(X,'fro')**2 + (lambda_y)*LA.norm(X,'fro')**2 )/b_norm**2)

        if i>=15:
            rel_error_kappa = (rel_error[i] / rel_error[i-15])**(1/15)

        if (rel_error[i] <= tol_res_error) or (rel_error_kappa > kappa)  :
            break
        else:
            Z = Res*mask + Z*mask_inv


    rel_error = rel_error[:it]
    return Z, it, rel_error, np.linalg.cond(Z)
