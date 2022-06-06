#
# LINMA2990 - Master Thesis
# The mathematics of Tennis Ranking: Game outcome prediction by low-rank methods
#
# Main code to run
# 
# Bastien Massion
# Created on 28/12/2021
#
# Last version on 14/04/2022
#


import numpy as np

#Import data
from Data_thesis import *

#Select data
from Selection_thesis import *
from Selection_tensor_thesis import *

#Manipulate data
from Statistics_thesis import *

#Verify priors
from Prior_thesis import *

#Solving optimization problem
from Optimization_thesis import *
from Optimization_levels_thesis import *

#Compute error metrics
from Error_thesis import *
from Error_tensor_thesis import *

#Plotting
from Plotting_thesis import *


# PARAMETERS TO CHANGE
testing_percentage = 30.0
#limit_dates = None
limit_dates = ['01/01/2016', '31/12/2016']

sorting_criterion = "activity"
#sorting_criterion = "wins"
#sorting_criterion = "losses"

number_players = 50     # max : 1246

estimator = "MAP"
#estimator = "conditional_mean"

alpha = 2.5
b = 2.0     # 0.35 * alpha**2 + 0.25

probability_model = ("uniform", None)
#probability_model = ("beta", b)
#probability_model = ("exponential_like", alpha)
#probability_model = ("logistic_like", alpha)

unknown_values = np.nan
diagonal_values = np.nan

imposed_rank = 2
tau_completion = 5.0
k = 2

#completion_method = ("SDP_NNM",)
#completion_method = ("SoftImpute", tau_completion)
#completion_method = ("iterative_SVD", imposed_rank)
#completion_method = ("lmafit", imposed_rank)
completion_method = ("LASSO", tau_completion)
#completion_method = ("WLASSO", tau_completion)
#completion_method = ("kNN", k)
#completion_method = ("columns_mean",)
#completion_method = ("rows_mean",)
#completion_method = ("half",)

tau_creation = 5.0

#creation_method = ("MAP_NNM", tau_creation)
#creation_method = ("MAP_BMF", imposed_rank)

#extra_dimension_name = 'Surface'
#extra_dimension_name = 'Location'
extra_dimension_name = 'Tournament'

imposed_lambda = 1.0
s = 1.0

#rating_model = ("uniform", None)
rating_model = ("logistic", s)

normalization = None
#normalization = "positive"
#normalization = "affine"
#normalization = "stochastic"

n_iter_global = 20
creation_method = ("Eratings", imposed_rank, imposed_lambda, normalization, n_iter_global)


in_range = True
add_to_one = True
diagonal_half = True
clipping = True


######## IMPORTING DATA FROM DATASET
#(data, header, data_types, data_values, data_categories, players, n_players, players_indices) = manageData('../Data/Data.csv')


######## VERIFYING PRIOR ASSUMPTIONS
#players_rankings, rankings, ranking_distribution, elo, elo_distribution = verifyPrior(data, players, header)


######## DATA SELECTION
#train_data, test_data, train_confrontation_matrix, test_confrontation_matrix, train_selected_omega, test_selected_omega, selected_players, selected_players_indices, selected_n_players = selectingData(data, players, players_indices, header, testing_percentage = testing_percentage, sorting_criterion = sorting_criterion, number_players = number_players, limit_dates = limit_dates)
train_data, test_data, train_confrontation_tensor, test_confrontation_tensor, train_selected_omega, test_selected_omega, selected_players, selected_players_indices, selected_n_players, selected_extra_dimension_values, selected_extra_dimension_indices, selected_n_extra_dimension = selectingDataTensor(data, players, players_indices, extra_dimension_name, header, data_values, testing_percentage = testing_percentage, sorting_criterion = sorting_criterion, number_players = number_players, limit_dates = limit_dates)


######## MATRIX COMPLETION EXPERIMENTS
"""
# TRAIN PROBABILITY MATRIX
p_bar_train = createProbabilityMatrix(train_confrontation_matrix,
                                      estimator = estimator, probability_model = probability_model, 
                                      unknown_values = unknown_values, diagonal_values = diagonal_values)


to_compare = [("SDP_NNM",), ("SoftImpute", 5.0), ("iterative_SVD", 2), ("lmafit", 2),
              ("kNN", 10), ("LASSO", 5.0), ("WLASSO", 5.0), ("columns_mean",), ("rows_mean",)]
#to_compare = [("SoftImpute", 5.0)]

#to_compare = np.linspace(1.0, 5.0, num = 50) # for b values
#to_compare = np.logspace(-4, 3, num = 50) # for tau values
#to_compare = np.arange(1, selected_n_players, dtype = int) # for r or k values

n_comparisons = len(to_compare)
#n_comparisons = 10
p_matrix = np.zeros((n_comparisons,2,selected_n_players,selected_n_players))
completion_time = np.zeros((n_comparisons,2))

# COMPUTATION OF THE MATRIX
for i in range(n_comparisons):
#    imposed_tau = to_compare[i]
#    completion_method = ("SoftImpute", imposed_tau)
    
#    imposed_rank = to_compare[i]
#    completion_method = ("iterative_SVD", imposed_rank)
    
#    imposed_k = to_compare[i]
#    completion_method = ("kNN", imposed_k)
    
#    imposed_b = to_compare[i]
#    probability_model = ("beta", imposed_b)
    
    completion_method = to_compare[i]
    
#    train_data, test_data, train_confrontation_matrix, test_confrontation_matrix, train_selected_omega, test_selected_omega, selected_players, selected_players_indices, selected_n_players = selectingData(data, players, players_indices, header, testing_percentage = testing_percentage, sorting_criterion = sorting_criterion, number_players = number_players, limit_dates = limit_dates)

    
    p_bar_train = createProbabilityMatrix(train_confrontation_matrix,
                                      estimator = estimator, probability_model = probability_model, 
                                      unknown_values = unknown_values, diagonal_values = diagonal_values)

    clipping = False
    p_matrix[i][0], completion_time[i][0] = complete(p_bar_train, train_selected_omega, train_confrontation_matrix,
                                         completion_method,
                                         in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
    
    clipping = True
    p_matrix[i][1] = clipToZeroOne(np.copy(p_matrix[i][0]))
    completion_time[i][1] = completion_time[i][0]


# PROBLEM INFORMATION
percentage_known_elements, n_unknown_elements = percentageKnownElements(train_selected_omega, selected_n_players, diagonal_values = diagonal_values)
    

# TEST PROBABILITY MATRIX
p_bar_test = createProbabilityMatrix(test_confrontation_matrix, 
                                     estimator = "MAP", probability_model = ("uniform", None), 
                                     unknown_values = unknown_values, diagonal_values = diagonal_values)

# ERROR TESTING
rmse, wrmse, singular_values, acc_mean, acc_std, acc_upper_bound, acc_known, out_of_range, not_adding_to_one, not_half_on_diagonal, rank = createMetricsMatrices(n_comparisons, selected_n_players)

for i in range(n_comparisons):
#    imposed_tau = to_compare[i]
#    completion_method = ("SoftImpute", imposed_tau)
    
#    imposed_rank = to_compare[i]
#    completion_method = ("iterative_SVD", imposed_rank)
    
#    imposed_k = to_compare[i]
#    completion_method = ("kNN", imposed_k)
    
    completion_method = to_compare[i]
    
#    imposed_b = to_compare[i]
#    probability_model = ("beta", imposed_b)
    
#    p_bar_test = createProbabilityMatrix(test_confrontation_matrix, 
#                                     estimator = "MAP", probability_model = probability_model, 
#                                     unknown_values = unknown_values, diagonal_values = diagonal_values)

    
    clipping = False
    rmse[i][0], wrmse[i][0], singular_values[i][0], acc_mean[i][0], acc_std[i][0], acc_upper_bound, acc_known, out_of_range[i][0], not_adding_to_one[i][0], not_half_on_diagonal[i][0], rank[i][0] = errorMetrics(p_matrix[i][0], p_bar_test, test_selected_omega, test_confrontation_matrix, completion_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
    
    clipping = True
    rmse[i][1], wrmse[i][1], singular_values[i][1], acc_mean[i][1], acc_std[i][1], acc_upper_bound, acc_known, out_of_range[i][1], not_adding_to_one[i][1], not_half_on_diagonal[i][1], rank[i][1] = errorMetrics(p_matrix[i][1], p_bar_test, test_selected_omega, test_confrontation_matrix, completion_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)

#plotVaryTau(to_compare, acc_mean, rmse, wrmse, completion_method[0])
#plotVaryRank(to_compare, acc_mean, rmse, wrmse, completion_method[0])
#plotVaryK(to_compare, acc_mean, rmse, wrmse, completion_method[0])
#plotVaryB(to_compare, acc_mean, rmse, wrmse, probability_model[0], estimator)
#plotVariabilityRuns(np.arange(1,n_comparisons +1, dtype = int), acc_mean, completion_method[0])
"""
####### MATRIX CREATION EXPERIMENTS
"""
#to_compare = np.logspace(-4, 3, num = 40)
#to_compare = np.arange(1, selected_n_players, dtype = int)
#to_compare = np.linspace(1.0, 5.0, num = 50) # for b values
to_compare = [("MAP_NNM", tau_creation)]

n_comparisons = len(to_compare)

p_matrix = np.zeros((n_comparisons,2,selected_n_players,selected_n_players))
completion_time = np.zeros((n_comparisons,2))

# COMPUTATION OF THE MATRIX
for i in range(n_comparisons):
#    imposed_b = to_compare[i]
#    probability_model = ("beta", imposed_b)

    creation_method = to_compare[i]
    
    clipping = False
    p_matrix[i][0], completion_time[i][0] = create(train_confrontation_matrix, creation_method, 
            estimator = estimator, probability_model = probability_model, 
            in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)

    clipping = True
    p_matrix[i][1] = clipToZeroOne(np.copy(p_matrix[i][0]))
    completion_time[i][1] = completion_time[i][0]


# ERROR TESTING
rmse, wrmse, singular_values, acc_mean_MAP_NNM, acc_std, acc_upper_bound, acc_known, out_of_range, not_adding_to_one, not_half_on_diagonal, rank = createMetricsMatrices(n_comparisons, selected_n_players)
p_bar_test = None

for i in range(n_comparisons):
#    imposed_rank = to_compare[i]
#    creation_method = ("MAP_BMF", imposed_rank)
    
#    imposed_b = to_compare[i]
#    probability_model = ("beta", imposed_b)
    
    clipping = False
    rmse[i][0], wrmse[i][0], singular_values[i][0], acc_mean_MAP_NNM[i][0], acc_std[i][0], acc_upper_bound, acc_known, out_of_range[i][0], not_adding_to_one[i][0], not_half_on_diagonal[i][0], rank[i][0] = errorMetrics(p_matrix[i][0], p_bar_test, test_selected_omega, test_confrontation_matrix, creation_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
    
    clipping = True
    rmse[i][1], wrmse[i][1], singular_values[i][1], acc_mean_MAP_NNM[i][1], acc_std[i][1], acc_upper_bound, acc_known, out_of_range[i][1], not_adding_to_one[i][1], not_half_on_diagonal[i][1], rank[i][1] = errorMetrics(p_matrix[i][1], p_bar_test, test_selected_omega, test_confrontation_matrix, creation_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)


#plotVaryTau(to_compare, acc_mean, rmse, wrmse, creation_method[0])
#plotVaryRank(to_compare, acc_mean, rmse, wrmse, creation_method[0])
#plotVaryB(to_compare, acc_mean, rmse, wrmse, probability_model[0], None)
"""

####### MATRIX CREATION LEVELS FORMULATION EXPERIMENTS

#to_compare = [("Eratings", imposed_rank, imposed_lambda, n_iter_global)]
#to_compare = np.logspace(-1.5, 2.5, num = 20) # for lambda values
#rank_max = selected_n_players//3
#to_compare = np.arange(1, rank_max+1, dtype = int)
to_compare = np.arange(1)

n_comparisons = len(to_compare)
p_tensor = np.zeros((n_comparisons,2,selected_n_players,selected_n_players,selected_n_extra_dimension))
e_matrix = np.zeros((n_comparisons,selected_n_players,selected_n_extra_dimension))
c_factor = np.zeros((n_comparisons,selected_n_players,imposed_rank))
t_factor = np.zeros((n_comparisons,selected_n_extra_dimension,imposed_rank))
completion_time = np.zeros((n_comparisons,2))

imposed_lambda = 1.0
creation_method = ("Eratings", imposed_rank, imposed_lambda, normalization, n_iter_global)
s = 10/(imposed_lambda)
rating_model = ("logistic", s)

# COMPUTATION OF THE MATRIX
for i in range(n_comparisons):   
#    imposed_lambda = to_compare[i]
#    creation_method = ("Eratings", imposed_rank, imposed_lambda, n_iter_global)
#    s = 15/(imposed_lambda)
#    rating_model = ("logistic", s)
    
#    imposed_s = to_compare[i]
#    rating_model = ("logistic", imposed_s)
#    imposed_lambda = 10/(imposed_s)
#    creation_method = ("Eratings", imposed_rank, imposed_lambda, n_iter_global)
    
#    imposed_rank = to_compare[i]
#    creation_method = ("Eratings", imposed_rank, imposed_lambda, normalization, n_iter_global)
#    
#    print("RANK", imposed_rank)
    
    clipping = False
    p_tensor[i][0], e_matrix[i], c_factor[i], t_factor[i], completion_time[i][0] = createTensor(train_confrontation_tensor, creation_method, 
            estimator = estimator, rating_model = rating_model, 
            in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)

    clipping = True
    p_tensor[i][1] = clipToZeroOneTensor(np.copy(p_tensor[i][0]))
    completion_time[i][1] = completion_time[i][0]

# ERROR TESTING
rmse, wrmse, singular_values, acc_mean, acc_std, acc_upper_bound, acc_known, out_of_range, not_adding_to_one, not_half_on_diagonal, rank = createMetricsMatricesTensor(n_comparisons, selected_n_players, selected_n_extra_dimension)
p_bar_test = None

for i in range(n_comparisons):
#    imposed_lambda = to_compare[i]
#    creation_method = ("Eratings", imposed_rank, imposed_lambda, n_iter_global)
#    s = 10/(imposed_lambda)
#    rating_model = ("logistic", s)
    
#    imposed_s = to_compare[i]
#    rating_model = ("logistic", imposed_s)
#    imposed_lambda = 10/(imposed_s)
#    creation_method = ("Eratings", imposed_rank, imposed_lambda, n_iter_global)
    
#    imposed_rank = to_compare[i]
#    creation_method = ("Eratings", imposed_rank, imposed_lambda, normalization, n_iter_global)
    
    clipping = False
    rmse[i][0], wrmse[i][0], singular_values[i][0], acc_mean[i][0], acc_std[i][0], acc_upper_bound, acc_known, out_of_range[i][0], not_adding_to_one[i][0], not_half_on_diagonal[i][0], rank[i][0] = errorMetricsTensor(p_tensor[i][0], e_matrix[i], p_bar_test, test_confrontation_tensor, creation_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
    
    clipping = True
    rmse[i][1], wrmse[i][1], singular_values[i][1], acc_mean[i][1], acc_std[i][1], acc_upper_bound, acc_known, out_of_range[i][1], not_adding_to_one[i][1], not_half_on_diagonal[i][1], rank[i][1] = errorMetricsTensor(p_tensor[i][1], e_matrix[i], p_bar_test, test_confrontation_tensor, creation_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)

#creation_method = ("Eratings", imposed_rank, imposed_lambda, n_iter_global)
#plotVaryLambda(to_compare, acc_mean, creation_method[0])
#plotVaryS(to_compare, acc_mean, creation_method[0])
#plotVaryRank(to_compare, acc_mean, rmse, wrmse, creation_method[0])


#e_matrix_mean, e_matrix_sigma = plotStdE(to_compare, e_matrix, creation_method[0])
