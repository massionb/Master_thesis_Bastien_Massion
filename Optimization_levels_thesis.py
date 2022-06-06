#
# LINMA2990 - Master Thesis
# The mathematics of Tennis Ranking: Game outcome prediction by low-rank methods
#
# Optimization for levels formulation
# 
# Bastien Massion
# Created on 15/04/2022
#
# Last version on 15/04/2022
#

from Statistics_thesis import *

#Manipulate data
import numpy as np

#Optimization process
import cvxpy as cp

#Measuring optimization time
import time


# OPTIMIZATION METHOD FOR LEVELS FORMULATION
    
def createTensor(train_confrontation_tensor, method, estimator = "MAP", rating_model = ("uniform", None), in_range = False, add_to_one = False, diagonal_half = False, clipping = True):
    if estimator != "MAP":
        print("Define an estimator acceptable for creation formulation")
        return
    method_name = method[0]
    before_time = time.time()
    if method_name == "Eratings":
        imposed_rank = method[1]
        imposed_lambda = method[2]
        normalization = method[3]
        n_iter_global = method[4]
        e_matrix, p_factor_matrix, t_factor_matrix = logPosteriorFactorization(train_confrontation_tensor, imposed_rank, imposed_lambda, rating_model, normalization, n_iter_global = n_iter_global)
    else :
        print("Define a correct creation method")
        return
    
    print("Creation method :", method_name)
    after_time = time.time()
    completion_time = after_time - before_time
    completion_time_hours = int(completion_time // 3600)
    completion_time_minutes = int((completion_time - completion_time_hours*3600) // 60)
    completion_time_seconds = completion_time - completion_time_hours*3600 - completion_time_minutes*60
    print("Completion time : %d h %d min %.3f sec" %(completion_time_hours, completion_time_minutes, completion_time_seconds))
    
    p_tensor = computeProbabilityTensor(e_matrix, imposed_lambda)
    
    if in_range == True:
        p_tensor = inRangeTensor(p_tensor)
    if add_to_one == True:
        p_tensor = addToOneTensor(p_tensor)
    if diagonal_half == True:
        p_tensor = halfOnDiagonalTensor(p_tensor)
    if clipping == True:
        p_tensor = clipToZeroOneTensor(p_tensor)
    
    return p_tensor, e_matrix, p_factor_matrix, t_factor_matrix, completion_time


def logPosteriorFactorization(train_confrontation_tensor, imposed_rank, imposed_lambda, rating_model, normalization = None, eps = 10**(-3), n_iter_global = 20, n_iter_local = 1000):
    n_players = len(train_confrontation_tensor)
    n_extra_dimension = len(train_confrontation_tensor[0,0,:])
    
    C_old = np.zeros((n_players, imposed_rank))
    T_old = np.zeros((n_extra_dimension, imposed_rank))
    E_old = C_old @ T_old.T
    
    C_new = np.random.normal(size = (n_players, imposed_rank))
    C_new = C_new - np.mean(C_new, axis = 0)
    T_new = np.random.normal(size = (n_extra_dimension, imposed_rank))
    E_new = C_new @ T_new.T
    
    delta = np.linalg.norm(E_new-E_old) + 2*eps
    
    it = 0
    print("Iteration : 0")
    print("Approximate error on E : ", delta)
    it += 1
    while delta > eps and it <= n_iter_global:
        
        ## REINITIALIZE MATRICES
        
        C_old = C_new
        T_old = T_new
        E_old = E_new
        
        
        ## COMPUTE P
        
        C = cp.Variable((n_players, imposed_rank))
    
        constraint_C = [np.ones(n_players) @ C == 0.0]
        
        obj_log_likelihood_C = 0.0
        for i in range(n_players):
            for j in range(n_players):
                for c in range(n_extra_dimension):
                    matches_ijc = train_confrontation_tensor[i,j,c]
                    if matches_ijc != 0 :
                        obj_log_likelihood_C += matches_ijc * cp.logistic(- imposed_lambda*(C[i,:]-C[j,:]) @ T_new[c,:] )
                    
        obj_log_prior_C = 0.0
        if rating_model[0] == "uniform":
            obj_log_prior_C = 0.0
        elif rating_model[0] == "logistic":
            s = rating_model[1]
            for i in range(n_players):
                for c in range(n_extra_dimension):
                    obj_log_prior_C += 1/s * ( C[i,:] @ T_new[c,:] ) + 2* cp.logistic(-1/s * (C[i,:] @ T_new[c,:]) )
        else:
            obj_log_prior_C = 10
            print("UNKNOWN RATING PRIOR MODEL")

        obj_C = cp.Minimize(obj_log_likelihood_C + obj_log_prior_C)
        
        prob_C = cp.Problem(obj_C, constraint_C)
        prob_C.solve(verbose=False, max_iters=n_iter_local)
        obj_value_C = prob_C.value
        C_new = C.value
        
        ## COMPUTE T
        
        T = cp.Variable((n_extra_dimension, imposed_rank))
        
        if normalization == None:
            constraint_T = []           
        elif normalization == "positive":
            constraint_T = [T >= 0.0]
        elif normalization == "affine":
            constraint_T = [T @ np.ones(imposed_rank) == 1.0]
        elif normalization == "stochastic":
            constraint_T1 = [T >= 0.0]
            constraint_T2 = [T @ np.ones(imposed_rank) == 1.0]
            constraint_T = constraint_T1 + constraint_T2
        else: 
            print("UNKNOWN NORMALIZATION")
            
        
        obj_log_likelihood_T = 0.0
        for i in range(n_players):
            for j in range(n_players):
                for c in range(n_extra_dimension):
                    matches_ijc = train_confrontation_tensor[i,j,c]
                    if matches_ijc != 0 :
                        obj_log_likelihood_T += matches_ijc * cp.logistic(- imposed_lambda*(C_new[i,:]-C_new[j,:]) @ T[c,:] )
        
        obj_log_prior_T = 0.0
        if rating_model[0] == "uniform":
            obj_log_prior_T = 0.0
        elif rating_model[0] == "logistic":
            s = rating_model[1]
            for i in range(n_players):
                for c in range(n_extra_dimension):
                    obj_log_prior_T += 1/s * ( C_new[i,:] @ T[c,:] ) + 2* cp.logistic(-1/s * (C_new[i,:] @ T[c,:]) )
        else: 
            obj_log_prior_T = 10
            print("UNKNOWN RATING PRIOR MODEL")
            
        obj_T = cp.Minimize(obj_log_likelihood_T + obj_log_prior_T)
        
        prob_T = cp.Problem(obj_T, constraint_T)
        prob_T.solve(verbose=False, max_iters=n_iter_local)
        obj_value_T = prob_T.value
        T_new = T.value
                
        ## COMPUTE E
        
        E_new = C_new @ T_new.T
        
        ## PREPARE NEXT ITERATION
        delta = np.linalg.norm(E_new-E_old)
        print("Iteration : ", it)
        print("Log-posterior objective (C block) : ", obj_value_C)
        print("Log-posterior objective (T block) : ", obj_value_T)
        print("Approximate error on E : ", delta)
        it += 1
    
    return E_new, C_new, T_new
    

def computeProbabilityTensor(e_matrix, imposed_lambda):
    n_players, n_extra_dimension = np.shape(e_matrix)
    probability_tensor = np.zeros((n_players, n_players, n_extra_dimension))
    
    for i in range(n_players):
        for j in range(n_players):
            for c in range(n_extra_dimension):
                probability_tensor[i,j,c] = 1.0 / (1.0 + np.exp(-imposed_lambda*(e_matrix[i,c] - e_matrix[j,c])) )
    
    return probability_tensor

# IMPOSING CONDITIONS

def inRangeTensor(p_tensor):
    n_players = len(p_tensor)
    n_extra_dimension = len(p_tensor[0,0,:])
    
    for i in range(n_players):
        for j in range(n_players):
            for c in range(n_extra_dimension):
                if p_tensor[i][j][c] < 0.0:
                    p_tensor[i][j][c] = 0.0
                elif p_tensor[i][j][c] > 1.0:
                    p_tensor[i][j][c] = 1.0
    return p_tensor

def addToOneTensor(p_tensor):
    n_players = len(p_tensor)
    n_extra_dimension = len(p_tensor[0,0,:])
    
    for i in range(n_players):
        for j in range(i+1, n_players):
            for c in range(n_extra_dimension):
                total_prob_ijc = p_tensor[i][j][c] + p_tensor[j][i][c]
                if total_prob_ijc == 0.0:
                    p_tensor[i][j][c] = 0.5
                    p_tensor[j][i][c] = 0.5
                elif total_prob_ijc != 1.0:
                    p_tensor[i][j][c] = p_tensor[i][j][c]/total_prob_ijc
                    p_tensor[j][i][c] = p_tensor[j][i][c]/total_prob_ijc           
    return p_tensor

def halfOnDiagonalTensor(p_tensor):
    n_players = len(p_tensor)
    n_extra_dimension = len(p_tensor[0,0,:])
    
    for i in range(n_players):
        for c in range(n_extra_dimension):
            p_tensor[i][i][c] = 0.5        
    return p_tensor

def clipToZeroOneTensor(p_tensor):
    n_players = len(p_tensor)
    n_extra_dimension = len(p_tensor[0,0,:])
    
    for i in range(n_players):
        for j in range(i+1, n_players):
            for c in range(n_extra_dimension):
                if p_tensor[j][i][c] < p_tensor[i][j][c]:
                    p_tensor[i][j][c] = 1.0
                    p_tensor[j][i][c] = 0.0
                elif p_tensor[j][i][c] > p_tensor[i][j][c]:
                    p_tensor[i][j][c] = 0.0
                    p_tensor[j][i][c] = 1.0
                else : 
                    p_tensor[i][j][c] = 0.5
                    p_tensor[j][i][c] = 0.5
    print("Probabilities clipped to 0 or to 1")
    return p_tensor



