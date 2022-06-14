#
# LINMA2990 - Master Thesis
# The mathematics of Tennis Ranking: Game outcome prediction by low-rank methods
#
# Experiment: Variability of runs with Matrix Completion (Chapter 2)
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
from Selection_thesis import *

#Manipulate data
from Statistics_thesis import *

#Solving optimization problem
from Optimization_thesis import *

#Compute error metrics
from Error_thesis import *

#Plotting
from Plotting_thesis import *

######## IMPORTING DATA FROM DATASET

(data, header, data_types, data_values, data_categories, players, n_players, players_indices) = manageData('Data.csv')


######## DATA SELECTION

number_players = 50
testing_percentage = 30.0
limit_dates = ['01/01/2013', '31/12/2013']
sorting_criterion = "activity"


######## MATRIX COMPLETION EXPERIMENT

# COMPLETION PARAMETERS
completion_method = ("LASSO", 5.0)
in_range = True
add_to_one = True
diagonal_half = True

# TRUE EXPERIMENT
n_runs = 4
to_compare = np.arange(1,n_runs+1)

n_comparisons = len(to_compare)
p_matrix = np.zeros((n_comparisons,2,selected_n_players,selected_n_players))
completion_time = np.zeros((n_comparisons,2))
rmse, wrmse, singular_values, acc_mean, acc_std, acc_upper_bound, acc_known, out_of_range, not_adding_to_one, not_half_on_diagonal, rank = createMetricsMatrices(n_comparisons, selected_n_players)


for i in range(n_comparisons):
    run_number = to_compare[i]
    
    train_data, test_data, train_confrontation_matrix, test_confrontation_matrix, train_selected_omega, test_selected_omega, selected_players, selected_players_indices, selected_n_players = selectingData(data, players, players_indices, header, testing_percentage = testing_percentage, sorting_criterion = sorting_criterion, number_players = number_players, limit_dates = limit_dates)
    
    # TRAIN PROBABILITY MATRIX
    p_bar_train = createProbabilityMatrix(train_confrontation_matrix,
                                          estimator = "MAP", probability_model = ("uniform", None), 
                                          unknown_values = np.nan, diagonal_values = np.nan)
    
    # TEST PROBABILITY MATRIX
    p_bar_test = createProbabilityMatrix(test_confrontation_matrix, 
                                         estimator = "MAP", probability_model = ("uniform", None), 
                                         unknown_values = np.nan, diagonal_values = np.nan)

    
    clipping = False
    p_matrix[i][0], completion_time[i][0] = complete(p_bar_train, train_selected_omega, train_confrontation_matrix,
                                         completion_method,
                                         in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
    
    clipping = True
    p_matrix[i][1] = clipToZeroOne(np.copy(p_matrix[i][0]))
    completion_time[i][1] = completion_time[i][0]

    # ERROR TESTING
    
    clipping = False
    rmse[i][0], wrmse[i][0], singular_values[i][0], acc_mean[i][0], acc_std[i][0], acc_upper_bound, acc_known, out_of_range[i][0], not_adding_to_one[i][0], not_half_on_diagonal[i][0], rank[i][0] = errorMetrics(p_matrix[i][0], p_bar_test, test_selected_omega, test_confrontation_matrix, completion_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
    
    clipping = True
    rmse[i][1], wrmse[i][1], singular_values[i][1], acc_mean[i][1], acc_std[i][1], acc_upper_bound, acc_known, out_of_range[i][1], not_adding_to_one[i][1], not_half_on_diagonal[i][1], rank[i][1] = errorMetrics(p_matrix[i][1], p_bar_test, test_selected_omega, test_confrontation_matrix, completion_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)

# PLOTTING 
plotVariabilityRuns(to_compare, acc_mean, completion_method[0])
    