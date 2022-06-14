#
# LINMA2990 - Master Thesis
# The mathematics of Tennis Ranking: Game outcome prediction by low-rank methods
#
# Experiment: Lambda.s Product for MAP on E (Chapter 4)
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

n_iter_global = 20
imposed_rank = 2
imposed_lambda = 10
imposed_s = 1


# NO TRAIN/TEST PROBABILITY MATRIX
p_bar_train = None
p_bar_test = None


# TRUE EXPERIMENT

to_compare = [None, "positive", "affine", "stochastic"]
n_comparisons = len(to_compare)
p_tensor = np.zeros((n_comparisons,2,selected_n_players,selected_n_players,selected_n_extra_dimension))
e_matrix = np.zeros((n_comparisons,selected_n_players,selected_n_extra_dimension))
c_factor = np.zeros((n_comparisons,selected_n_players,imposed_rank))
t_factor = np.zeros((n_comparisons,selected_n_extra_dimension,imposed_rank))
completion_time = np.zeros((n_comparisons,2))

for i in range(n_comparisons):
    normalization = to_compare[i]
    
    creation_method = ("Eratings", imposed_rank, imposed_lambda, normalization, n_iter_global)
    rating_model = ("logistic", imposed_s)
        
    clipping = False
    p_tensor[i][0], e_matrix[i], c_factor[i], t_factor[i], completion_time[i][0] = createTensor(train_confrontation_tensor, creation_method, 
            rating_model = rating_model, 
            in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)

    clipping = True
    p_tensor[i][1] = clipToZeroOneTensor(np.copy(p_tensor[i][0]))
    completion_time[i][1] = completion_time[i][0]

# ERROR TESTING

rmse, wrmse, singular_values, acc_mean, acc_std, acc_upper_bound, acc_known, out_of_range, not_adding_to_one, not_half_on_diagonal, rank = createMetricsMatricesTensor(n_comparisons, selected_n_players, selected_n_extra_dimension)

for i in range(n_comparisons):

    clipping = False
    rmse[i][0], wrmse[i][0], singular_values[i][0], acc_mean[i][0], acc_std[i][0], acc_upper_bound, acc_known, out_of_range[i][0], not_adding_to_one[i][0], not_half_on_diagonal[i][0], rank[i][0] = errorMetricsTensor(p_tensor[i][0], e_matrix[i], p_bar_test, test_confrontation_tensor, creation_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
    
    clipping = True
    rmse[i][1], wrmse[i][1], singular_values[i][1], acc_mean[i][1], acc_std[i][1], acc_upper_bound, acc_known, out_of_range[i][1], not_adding_to_one[i][1], not_half_on_diagonal[i][1], rank[i][1] = errorMetricsTensor(p_tensor[i][1], e_matrix[i], p_bar_test, test_confrontation_tensor, creation_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)

