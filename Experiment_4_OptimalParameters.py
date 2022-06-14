#
# LINMA2990 - Master Thesis
# The mathematics of Tennis Ranking: Game outcome prediction by low-rank methods
#
# Experiment: Optimal Parameters for MAP on E (Chapter 4)
# 
# Bastien Massion
# Created on 06/06/2022
#
# Last version on 06/06/2022
#

import numpy as np

#Import data
from Data_thesis import *

#Select data
from Selection_tensor_thesis import *

#Manipulate data
from Statistics_thesis import *

#Solving optimization problem
from Optimization_levels_thesis import *

#Compute error metrics
from Error_tensor_thesis import *

#Plotting
from Plotting_thesis import *


######## IMPORTING DATA FROM DATASET

(data, header, data_types, data_values, data_categories, players, n_players, players_indices) = manageData('../Data/Data.csv')


######## DATA SELECTION

extra_dimension_name = 'Tournament'
number_players = 50
testing_percentage = 30.0
limit_dates = ['01/01/2013', '31/12/2013']
sorting_criterion = "activity"

train_data, test_data, train_confrontation_tensor, test_confrontation_tensor, train_selected_omega, test_selected_omega, selected_players, selected_players_indices, selected_n_players, selected_extra_dimension_values, selected_extra_dimension_indices, selected_n_extra_dimension = selectingDataTensor(data, players, players_indices, extra_dimension_name, header, data_values, testing_percentage = testing_percentage, sorting_criterion = sorting_criterion, number_players = number_players, limit_dates = limit_dates)


######## MATRIX COMPLETION EXPERIMENT

# COMPLETION PARAMETERS
in_range = True
add_to_one = True
diagonal_half = True

normalization = None
n_iter_global = 20
imposed_rank = 2


# NO TRAIN/TEST PROBABILITY MATRIX
p_bar_train = None
p_bar_test = None


# TRUE EXPERIMENT
to_compare = ["lambda","s","rank"]
n_comparisons = len(to_compare)

# COMPUTATION OF THE MATRIX
for i in range(n_comparisons):   
    parameter_name = to_compare[i]
    
    if parameter_name == "lambda":
        parameters_values = np.logspace(-1.5, 2.5, num = 20)
    elif parameter_name == "s":
        parameters_values = np.logspace(-1.5, 2.5, num = 20)
    elif parameter_name == "rank":
        rank_max = selected_n_players//3
        parameters_values = np.arange(1, rank_max+1, dtype = int)
        imposed_rank = rank_max
    else:
        print("Wrong parameter name")
        
        
    n_parameters = len(parameters_values)
    p_tensor = np.zeros((n_parameters,2,selected_n_players,selected_n_players,selected_n_extra_dimension))
    e_matrix = np.zeros((n_parameters,selected_n_players,selected_n_extra_dimension))
    c_factor = np.zeros((n_parameters,selected_n_players,imposed_rank))
    t_factor = np.zeros((n_parameters,selected_n_extra_dimension,imposed_rank))
    completion_time = np.zeros((n_parameters,2))
    
    for j in range(n_parameters):
        
        if parameter_name == "lambda":
            imposed_lambda = parameters_values[j]
            creation_method = ("Eratings", 2, imposed_lambda, normalization, n_iter_global)
            rating_model = ("logistic", 1)
        elif parameter_name == "s":
            imposed_s = parameters_values[j]
            creation_method = ("Eratings", 2, 1, normalization, n_iter_global)
            rating_model = ("logistic", imposed_s)
        elif parameter_name == "rank":
            imposed_rank = parameters_values[j]
            creation_method = ("Eratings", imposed_rank, 10, normalization, n_iter_global)
            rating_model = ("logistic", 1)
        else:
            print("Wrong parameter name")
        
        clipping = False
        p_tensor[j][0], e_matrix[j], c_factor[j,:,:imposed_rank], t_factor[j,:,:imposed_rank], completion_time[j][0] = createTensor(train_confrontation_tensor, creation_method, 
                                                                                                    rating_model = rating_model, 
                                                                                                    in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)

        clipping = True
        p_tensor[j][1] = clipToZeroOneTensor(np.copy(p_tensor[j][0]))
        completion_time[j][1] = completion_time[j][0]
        
    # ERROR TESTING
    rmse, wrmse, singular_values, acc_mean, acc_std, acc_upper_bound, acc_known, out_of_range, not_adding_to_one, not_half_on_diagonal, rank = createMetricsMatricesTensor(n_parameters, selected_n_players, selected_n_extra_dimension)
    
    for j in range(n_parameters):
        clipping = False
        rmse[j][0], wrmse[j][0], singular_values[j][0], acc_mean[j][0], acc_std[j][0], acc_upper_bound, acc_known, out_of_range[j][0], not_adding_to_one[j][0], not_half_on_diagonal[j][0], rank[j][0] = errorMetricsTensor(p_tensor[j][0], e_matrix[j], p_bar_test, test_confrontation_tensor, creation_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
        
        clipping = True
        rmse[j][1], wrmse[j][1], singular_values[j][1], acc_mean[j][1], acc_std[j][1], acc_upper_bound, acc_known, out_of_range[j][1], not_adding_to_one[j][1], not_half_on_diagonal[j][1], rank[j][1] = errorMetricsTensor(p_tensor[j][1], e_matrix[j], p_bar_test, test_confrontation_tensor, creation_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
        
    if parameter_name == "lambda":
        plotVaryLambda(parameters_values, acc_mean, "Eratings")
    elif parameter_name == "s":
        plotVaryS(parameters_values, acc_mean, "Eratings")
    elif parameter_name == "rank":
        plotVaryRank(parameters_values, acc_mean, rmse, wrmse,"Eratings")
    else:
        print("Wrong parameter name")
    
    