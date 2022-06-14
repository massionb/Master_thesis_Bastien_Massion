#
# LINMA2990 - Master Thesis
# The mathematics of Tennis Ranking: Game outcome prediction by low-rank methods
#
# Experiment: Prior distribution for MAP on P (Chapter 3)
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
creation_method = ("MAP_NNM", 5.0)
in_range = True
add_to_one = True
diagonal_half = True

# NO TRAIN/TEST PROBABILITY MATRIX
p_bar_train = None
p_bar_test = None


# TRUE EXPERIMENT
estimators = None
to_compare = ["beta"]
n_comparisons = len(to_compare)


for i in range(n_comparisons):
    
    prior = to_compare[i]
    # COMPUTATION OF THE MATRIX
    
    if prior == "beta":
        parameters_values = np.linspace(1.0, 5.0, num = 50) # for b values    
    else:
        print("Wrong distribution name")
        
    n_parameters = len(parameters_values)
    p_matrix = np.zeros((n_parameters,2,selected_n_players,selected_n_players))
    completion_time = np.zeros((n_parameters,2))
    
    for j in range(n_parameters):
        probability_model = (prior, parameters_values[j])
        
        clipping = False
        p_matrix[j][0], completion_time[j][0] = create(train_confrontation_matrix, creation_method, 
                                                        estimator = "MAP", probability_model = probability_model, 
                                                        in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
    
        clipping = True
        p_matrix[j][1] = clipToZeroOne(np.copy(p_matrix[j][0]))
        completion_time[j][1] = completion_time[j][0]
        

    # ERROR TESTING
    rmse, wrmse, singular_values, acc_mean, acc_std, acc_upper_bound, acc_known, out_of_range, not_adding_to_one, not_half_on_diagonal, rank = createMetricsMatrices(n_parameters, selected_n_players)
    
    for j in range(n_parameters):
                
        clipping = False
        rmse[j][0], wrmse[j][0], singular_values[j][0], acc_mean[j][0], acc_std[j][0], acc_upper_bound, acc_known, out_of_range[j][0], not_adding_to_one[j][0], not_half_on_diagonal[j][0], rank[j][0] = errorMetrics(p_matrix[j][0], p_bar_test, test_selected_omega, test_confrontation_matrix, creation_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
        
        clipping = True
        rmse[j][1], wrmse[j][1], singular_values[j][1], acc_mean[j][1], acc_std[j][1], acc_upper_bound, acc_known, out_of_range[j][1], not_adding_to_one[j][1], not_half_on_diagonal[j][1], rank[j][1] = errorMetrics(p_matrix[j][1], p_bar_test, test_selected_omega, test_confrontation_matrix, creation_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
        
        
    # PLOTTING    
    plotVaryB(parameters_values, acc_mean, rmse, wrmse, "beta", estimators)
    
    