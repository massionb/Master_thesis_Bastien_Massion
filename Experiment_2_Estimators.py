#
# LINMA2990 - Master Thesis
# The mathematics of Tennis Ranking: Game outcome prediction by low-rank methods
#
# Experiment: Estimators for Matrix Completion (Chapter 2)
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

train_data, test_data, train_confrontation_matrix, test_confrontation_matrix, train_selected_omega, test_selected_omega, selected_players, selected_players_indices, selected_n_players = selectingData(data, players, players_indices, header, testing_percentage = testing_percentage, sorting_criterion = sorting_criterion, number_players = number_players, limit_dates = limit_dates)


######## MATRIX COMPLETION EXPERIMENT

# COMPLETION PARAMETERS
completion_method = ("LASSO", 5.0)
in_range = True
add_to_one = True
diagonal_half = True

# TRUE EXPERIMENT
to_compare = [("MAP", "MAP"), ("CM", "CM"), ("MAP", "ML"), ("CM", "ML")]
n_comparisons = len(to_compare)


for i in range(n_comparisons):
    
    estimators = to_compare[i]
    
    # COMPUTATION OF THE MATRIX
    estimator_train = estimators[0]

    parameters_values = np.linspace(1.0, 5.0, num = 50) # for b values    

    n_parameters = len(parameters_values)
    p_matrix = np.zeros((n_parameters,2,selected_n_players,selected_n_players))
    completion_time = np.zeros((n_parameters,2))
    
    for j in range(n_parameters):
        probability_model = ("beta", parameters_values[j])
        
        # TRAIN PROBABILITY MATRIX
        p_bar_train = createProbabilityMatrix(train_confrontation_matrix,
                                              estimator = estimator_train, probability_model = probability_model, 
                                              unknown_values = np.nan, diagonal_values = np.nan)

        clipping = False
        p_matrix[j][0], completion_time[j][0] = complete(p_bar_train, train_selected_omega, train_confrontation_matrix,
                                             completion_method,
                                             in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
        
        clipping = True
        p_matrix[j][1] = clipToZeroOne(np.copy(p_matrix[j][0]))
        completion_time[j][1] = completion_time[j][0]
        

    # ERROR TESTING
    rmse, wrmse, singular_values, acc_mean, acc_std, acc_upper_bound, acc_known, out_of_range, not_adding_to_one, not_half_on_diagonal, rank = createMetricsMatrices(n_parameters, selected_n_players)
    
    estimator_test = estimators[1]
    
    for j in range(n_parameters):
        
        if estimator_test == "ML":
            estimator_test_bis = "MAP"
            probability_model = ("uniform", None)
        elif estimator_test == "MAP":
            estimator_test_bis = "MAP"
            probability_model = ("beta", parameters_values[j])
        elif estimator_test == "CM":
            estimator_test_bis = "CM"
            probability_model = ("beta", parameters_values[j])
        else:
            print("Wrong estimator name")
        
        # TEST PROBABILITY MATRIX
        p_bar_test = createProbabilityMatrix(test_confrontation_matrix, 
                                             estimator = estimator_test_bis, probability_model = probability_model, 
                                             unknown_values = np.nan, diagonal_values = np.nan)

    
        clipping = False
        rmse[j][0], wrmse[j][0], singular_values[j][0], acc_mean[j][0], acc_std[j][0], acc_upper_bound, acc_known, out_of_range[j][0], not_adding_to_one[j][0], not_half_on_diagonal[j][0], rank[j][0] = errorMetrics(p_matrix[j][0], p_bar_test, test_selected_omega, test_confrontation_matrix, completion_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
        
        clipping = True
        rmse[j][1], wrmse[j][1], singular_values[j][1], acc_mean[j][1], acc_std[j][1], acc_upper_bound, acc_known, out_of_range[j][1], not_adding_to_one[j][1], not_half_on_diagonal[j][1], rank[j][1] = errorMetrics(p_matrix[j][1], p_bar_test, test_selected_omega, test_confrontation_matrix, completion_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
        
        
    # PLOTTING    
    plotVaryB(parameters_values, acc_mean, rmse, wrmse, "beta", estimators)
    
    
    