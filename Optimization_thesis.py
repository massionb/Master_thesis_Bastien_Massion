#
# LINMA2990 - Master Thesis
# The mathematics of Tennis Ranking: Game outcome prediction by low-rank methods
#
# Optimization for probability formulation
# 
# Bastien Massion
# Created on 01/03/2022
#
# Last version on 23/05/2022
#


from Statistics_thesis import *
from lmafit import *

#Manipulate data
import numpy as np
import random

#Optimization process
from fancyimpute import NuclearNormMinimization, SoftImpute, KNN, IterativeSVD
import cvxpy as cp

#Measuring optimization time
import time


# OPTIMIZATION METHOD FOR MATRIX COMPLETION FORMULATION
    
def complete(p_bar_train, omega_train, train_confrontation_matrix, method, in_range = True, add_to_one = True, diagonal_half = True, clipping = True):
    method_name = method[0]
    before_time = time.time()
    if method_name == "SDP_NNM":
        p_matrix = nuclearNormMinimization(p_bar_train)
    elif method_name == "SoftImpute":
        tau = method[1]
        p_matrix = softImpute(p_bar_train, tau = tau)
    elif method_name == "iterative_SVD": 
        imposed_rank = method[1]
        p_matrix = iterativeSVD(p_bar_train, imposed_rank)
    elif method_name == "lmafit":
        imposed_rank = method[1]
        p_bar_train_proj, mask = omegaProjector(p_bar_train, omega_train, binary_mask = True)
        p_matrix, it, rel_error, cond = lmafit(p_bar_train_proj, mask, imposed_rank)
    elif method_name == "LASSO":
        tau = method[1]
        p_matrix = nuclearNormFrobeniusMinimization(p_bar_train, omega_train, tau_val = tau)
    elif method_name == "WLASSO":
        tau = method[1]
        p_matrix = nuclearNormFrobeniusMinimization(p_bar_train, omega_train, has_weights = True, weights = train_confrontation_matrix, tau_val = tau)
    elif method_name == "kNN":
        k = method[1]
        p_matrix = kNearestNeighbours(p_bar_train, k)
    elif method_name == "columns_mean":
        p_matrix = columnsMean(p_bar_train)
    elif method_name == "rows_mean":
        p_matrix = rowsMean(p_bar_train)
    elif method_name == "half":
        p_matrix = half(p_bar_train)
    else :
        print("Define a correct completion method")
        return
    
    print("Completion method :", method_name)

    if method_name == "SoftImpute" or method_name == "LASSO" or method_name == "WLASSO":
        print("tau :", tau)
    elif method_name == "lmafit" or method_name == "iterative_SVD":
        print("Imposed rank :", imposed_rank)
    elif method_name == "kNN":
        print("k :", k)
    after_time = time.time()
    completion_time = after_time - before_time
    completion_time_hours = int(completion_time // 3600)
    completion_time_minutes = int((completion_time - completion_time_hours*3600) // 60)
    completion_time_seconds = completion_time - completion_time_hours*3600 - completion_time_minutes*60
    print("Completion time : %d h %d min %.3f sec" %(completion_time_hours, completion_time_minutes, completion_time_seconds))
    
    if in_range == True:
        p_matrix = inRange(p_matrix)
    if add_to_one == True:
        p_matrix = addToOne(p_matrix)
    if diagonal_half == True:
        p_matrix = halfOnDiagonal(p_matrix)
    if clipping == True:
        p_matrix = clipToZeroOne(p_matrix)
        
    return p_matrix, completion_time

# Super slow, respects probability-range constraints, doesn't respect probability-sum constraints 
def nuclearNormMinimization(p_bar_train):
    p_matrix = NuclearNormMinimization(verbose = True).fit_transform(p_bar_train)
    return p_matrix


# Bad with BiScaler normalization (destroys probability-range constraints)
# Good without BiScaler normalization
# Fast, respects probability-range constraints, doesn't respect probability-sum constraints 
def softImpute(p_bar_train, tau = 5.0, max_iters = 100):
    p_matrix = SoftImpute(verbose = True, shrinkage_value = tau, max_iters = max_iters).fit_transform(p_bar_train)
    return p_matrix


# Fast, doesn't respect some probability-range constraints (sometimes badly), doesn't respect probability-sum constraints 
def iterativeSVD(p_bar_train, imposed_rank = 2):
    p_matrix = IterativeSVD(verbose = True, rank = imposed_rank, gradual_rank_increase = False).fit_transform(p_bar_train)
    return p_matrix


# Fast, for n = 50 converges only when tau >= 0.01, best with tau = 0.1
# Big tau : more importance to nuclear norm
# Small tau : less importance to nuclear norm
def nuclearNormFrobeniusMinimization(p_bar_train, omega, weights = None, has_weights = False, tau_val = 5.0, max_iters = 2000):
    p_bar_train_projected, binary_mask = omegaProjector(p_bar_train, omega, binary_mask = True)
    
    n_players = len(p_bar_train)
    P = cp.Variable((n_players,n_players))
    
    constraint1 = [P >= 0.0]
    constraint2 = [P <= 1.0]
    constraint3 = [P + P.T == 1.0]
    constraints = constraint1 + constraint2 + constraint3
    
    obj1 = cp.norm(P, "nuc")
    if has_weights == False:
        obj2 = cp.norm(cp.multiply(P,binary_mask) - p_bar_train_projected, "fro")**2
    else:
        weights_sqrt = np.sqrt(weights)
        obj2 = cp.norm(cp.multiply(weights_sqrt, cp.multiply(P,binary_mask) - p_bar_train_projected), "fro")**2
        
    if tau_val <= 1.0 :
        tau = cp.Parameter(nonneg=True)
        tau.value = tau_val
        obj = cp.Minimize(tau*obj1 + obj2)
    else :
        tau_inv = cp.Parameter(nonneg=True)
        tau_inv.value = 1.0/tau_val
        obj = cp.Minimize(obj1 + tau_inv*obj2)
        
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=False, max_iters=max_iters)
    obj_value = prob.value
    p_matrix = P.value
    return p_matrix

def kNearestNeighbours(p_bar_train, k):
    p_matrix = KNN(k=k).fit_transform(p_bar_train)
    return p_matrix

def columnsMean(p_bar_train):
    columnsMean = np.nanmean(p_bar_train, axis = 0)
    p_matrix = np.copy(p_bar_train)
    binary_mask = np.isnan(p_matrix)
    nan_indices = np.argwhere(binary_mask)
    for k in range(len(nan_indices)):
        i = nan_indices[k,0]
        j = nan_indices[k,1]
        p_matrix[i,j] = columnsMean[j]
    return p_matrix

def rowsMean(p_bar_train):
    rowsMean = np.nanmean(p_bar_train, axis = 1)
    p_matrix = np.copy(p_bar_train)
    binary_mask = np.isnan(p_matrix)
    nan_indices = np.argwhere(binary_mask)
    for k in range(len(nan_indices)):
        i = nan_indices[k,0]
        j = nan_indices[k,1]
        p_matrix[i,j] = rowsMean[i]
    return p_matrix

def half(p_bar_train):
    n_players = len(p_bar_train)
    p_matrix = np.ones((n_players, n_players))*0.5
    return p_matrix


# OPTIMIZATION METHODS FOR LOG-POSTERIOR FORMULATION
    
def create(train_confrontation_matrix, method, estimator = "MAP", probability_model = ("uniform", None), in_range = True, add_to_one = True, diagonal_half = True, clipping = True):
    if estimator != "MAP":
        print("Define an estimator acceptable for optimization formulation")
        return
    method_name = method[0]
    before_time = time.time()
    if method_name == "MAP_NNM":
        tau = method[1]
        p_matrix = MAPNNM(train_confrontation_matrix, probability_model, tau)
    elif method_name == "MAP_BMF":
        rank = method[1]
        p_matrix, u_factor_matrix, v_factor_matrix = MAPBMF(train_confrontation_matrix, rank, probability_model)
    else :
        print("Define a correct creation method")
        return
    
    print("Creation method :", method_name)
    after_time = time.time()
    completion_time = after_time - before_time
    print("Completion time : %.3f sec" %completion_time)
    
    if in_range == True:
        p_matrix = inRange(p_matrix)
    if add_to_one == True:
        p_matrix = addToOne(p_matrix)
    if diagonal_half == True:
        p_matrix = halfOnDiagonal(p_matrix)
    if clipping == True:
        p_matrix = clipToZeroOne(p_matrix)
    
    return p_matrix, completion_time


def MAPNNM(train_confrontation_matrix, probability_model, tau_val = 5.0, max_iters = 2000):
    n_players = len(train_confrontation_matrix)
    P = cp.Variable((n_players,n_players))
    
    constraint1 = [P >= 0.0]
    constraint2 = [P <= 1.0]
    constraint3 = [P + P.T == 1.0]
    constraints = constraint1 + constraint2 + constraint3
    
    obj_nuclear_norm = cp.norm(P, "nuc")
    obj_log_likelihood = - cp.sum( cp.multiply(train_confrontation_matrix, cp.log(P)) ) #- cp.sum( cp.multiply(train_confrontation_matrix.T, cp.log(1.0 - P)) )
    obj_log_prior = 0.0
    
    if probability_model[0] != "uniform":
        obj_log_prior = 10 # TO CHANGE
        print("NON UNIFORM PRIOR")
        
    if tau_val <= 1.0 :
        tau = cp.Parameter(nonneg=True)
        tau.value = tau_val
        obj = cp.Minimize(tau*obj_nuclear_norm + obj_log_likelihood + obj_log_prior)
    else :
        tau_inv = cp.Parameter(nonneg=True)
        tau_inv.value = 1.0/tau_val
        obj = cp.Minimize(obj_nuclear_norm + tau_inv*(obj_log_likelihood + obj_log_prior))
    
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=True, max_iters=max_iters)
    obj_value = prob.value
    p_matrix = P.value
    return p_matrix
    

def MAPBMF(train_confrontation_matrix, imposed_rank, probability_model, eps = 10**(-3), n_iter_global = 20, n_iter_local = 1000):
  
    n_players = len(train_confrontation_matrix)
    
    U_old = np.zeros((n_players, imposed_rank))
    V_old = np.zeros((n_players, imposed_rank))
    P_old = U_old @ V_old.T
    
    if imposed_rank == 1:
        U_new = np.random.uniform(size = (n_players, imposed_rank))
        V_new = np.ones((n_players, imposed_rank))
    else:
        U_new = np.random.uniform(size = (n_players, imposed_rank))
        V_new = np.random.uniform(size = (n_players, imposed_rank))
        
    P_new = U_new @ V_new.T
    
    delta = np.linalg.norm(P_new-P_old) + 2*eps
    
    it = 0
    print("Iteration : 0")
    print("Approximate error on P : ", delta)
    it += 1

    while delta > eps and it <= n_iter_global:
        
        ## REINITIALIZE MATRICES
        
        U_old = U_new
        V_old = V_new
        P_old = P_new
        
        
        ## COMPUTE U
        
        U = cp.Variable((n_players, imposed_rank))
            
        constraint_U1 = [U @ V_new.T >= 0.0]
        constraint_U2 = [U @ V_new.T <= 1.0]
        constraint_U3 = [U @ V_new.T + V_new @ U.T == 1.0]
#        constraint_U3 = []
        constraints_U = constraint_U1 + constraint_U2 + constraint_U3
        
        obj_log_likelihood_U = - cp.sum( cp.multiply(train_confrontation_matrix, cp.log(U @ V_new.T)) ) #- cp.sum( cp.multiply(train_confrontation_matrix.T, cp.log(1.0 - P)) )

        obj_log_prior_U = 0.0
        if probability_model[0] == "uniform":
            obj_log_prior_U = 0.0
        elif probability_model[0] == "logistic_like":
            obj_log_prior_U = 0.0 #TO MODIFY
        else:
            obj_log_prior_U = 10
            print("UNKNOWN RATING PRIOR MODEL")

        obj_U = cp.Minimize(obj_log_likelihood_U + obj_log_prior_U)
        
        prob_U = cp.Problem(obj_U, constraints_U)
        prob_U.solve(verbose=False, max_iters=n_iter_local)
        obj_value_U = prob_U.value
        U_new = U.value
        
        
        ## COMPUTE V
        
        V = cp.Variable((n_players, imposed_rank))
        
        constraint_V1 = [(U_new @ V.T) >= 0.0]
        constraint_V2 = [U_new @ V.T <= 1.0]
        constraint_V3 = [U_new @ V.T + V @ U_new.T == 1.0]
        constraints_V = constraint_V1 + constraint_V2 + constraint_V3
        
        obj_log_likelihood_V = - cp.sum( cp.multiply(train_confrontation_matrix, cp.log(U_new @ V.T)) ) #- cp.sum( cp.multiply(train_confrontation_matrix.T, cp.log(1.0 - P)) )

        obj_log_prior_V = 0.0
        if probability_model[0] == "uniform":
            obj_log_prior_V = 0.0
        elif probability_model[0] == "logistic_like":
            obj_log_prior_V = 0.0 #TO MODIFY
        else:
            obj_log_prior_V = 10
            print("UNKNOWN RATING PRIOR MODEL")

        obj_V = cp.Minimize(obj_log_likelihood_V + obj_log_prior_V)
        
        prob_V = cp.Problem(obj_V, constraints_V)
        prob_V.solve(verbose=False, max_iters=n_iter_local)
        obj_value_V = prob_V.value
        V_new = V.value
        
        
        ## COMPUTE E
        
        P_new = U_new @ V_new.T
        
        
        ## PREPARE NEXT ITERATION
        delta = np.linalg.norm(P_new-P_old)
        print("Iteration : ", it)
        print("Log-posterior objective (U block) : ", obj_value_U)
        print("Log-posterior objective (V block) : ", obj_value_V)
        print("Approximate error on P : ", delta)
        it += 1
    
    return P_new, U_new, V_new


# IMPOSING CONDITIONS

def inRange(p_matrix):
    n_players = len(p_matrix)
    for i in range(n_players):
        for j in range(n_players):
            if p_matrix[i][j] < 0.0:
                p_matrix[i][j] = 0.0
            elif p_matrix[i][j] > 1.0:
                p_matrix[i][j] = 1.0
    return p_matrix

def addToOne(p_matrix):
    n_players = len(p_matrix)
    for i in range(n_players):
        for j in range(i+1, n_players):
            total_prob_ij = p_matrix[i][j] + p_matrix[j][i]
            if total_prob_ij == 0.0:
                p_matrix[i][j] = 0.5
                p_matrix[j][i] = 0.5
            elif total_prob_ij != 1.0:
                p_matrix[i][j] = p_matrix[i][j]/total_prob_ij
                p_matrix[j][i] = p_matrix[j][i]/total_prob_ij            
    return p_matrix

def halfOnDiagonal(p_matrix):
    n_players = len(p_matrix)
    for i in range(n_players):
        p_matrix[i][i] = 0.5        
    return p_matrix

def clipToZeroOne(p_matrix):
    n_players = len(p_matrix)
    for i in range(n_players):
        for j in range(i+1, n_players):
            if p_matrix[j][i] < p_matrix[i][j]:
                p_matrix[i][j] = 1.0
                p_matrix[j][i] = 0.0
            elif p_matrix[j][i] > p_matrix[i][j]:
                p_matrix[i][j] = 0.0
                p_matrix[j][i] = 1.0
            else : 
                p_matrix[i][j] = 0.5
                p_matrix[j][i] = 0.5
    print("Probabilities clipped to 0 or to 1")
    return p_matrix