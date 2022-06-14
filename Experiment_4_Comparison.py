#
# LINMA2990 - Master Thesis
# The mathematics of Tennis Ranking: Game outcome prediction by low-rank methods
#
# Experiment: Comparison between all methods (Chapter 4)
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
from Selection_tensor_thesis import *

#Manipulate data
from Statistics_thesis import *

#Solving optimization problem
from Optimization_thesis import *
from Optimization_levels_thesis import *

#Compute error metrics
from Error_thesis import *
from Error_tensor_thesis import *

#Plotting
from Plotting_thesis import *


######## IMPORTING DATA FROM DATASET

(data, header, data_types, data_values, data_categories, players, n_players, players_indices) = manageData('../Data/Data.csv')


######## DATA SELECTION

extra_dimension_name = 'Tournament'
number_players = 50
testing_percentage = 30.0
sorting_criterion = "activity"


######## COMPARISON EXPERIMENT

# CONSTRAINTS PARAMETERS

in_range = True
add_to_one = True
diagonal_half = True
clipping = True

probability_model = ("uniform", None)
rating_model = ("logistic", 1)

# METHODS TO COMPARE

to_compare_completion = [("SDP_NNM",), ("SoftImpute", 5.0), ("iterative_SVD", 2), ("lmafit", 2),
              ("kNN", 10), ("LASSO", 5.0), ("WLASSO", 5.0), ("columns_mean",), ("rows_mean",)]
to_compare_MAP_P = [("MAP_NNM", 5.0)]
to_compare_MAP_E = [("Eratings", 2, 10, None, 20)]
to_compare = to_compare_completion + to_compare_MAP_P + to_compare_MAP_E


n_comparisons_completion = len(to_compare_completion)
n_comparisons_MAP_P = len(to_compare_MAP_P)
n_comparisons_MAP_E = len(to_compare_MAP_E)
n_comparisons = len(to_compare)

# SEASONS TO COMPARE ON

seasons = ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016']
n_seasons = len(seasons)

completion_time = np.zeros((n_comparisons, n_seasons))
accuracy = np.zeros((n_comparisons, n_seasons))

for i in range(n_seasons):
    season = seasons[i]
    limit_dates = ['01/01/' + season, '31/12/' + season]
    
    j_tot = 0
    
    ######## MATRIX COMPLETION TECHNIQUES ########
    
    train_data, test_data, train_confrontation_matrix, test_confrontation_matrix, train_selected_omega, test_selected_omega, selected_players, selected_players_indices, selected_n_players = selectingData(data, players, players_indices, header, testing_percentage = testing_percentage, sorting_criterion = sorting_criterion, number_players = number_players, limit_dates = limit_dates)
    
    # TRAIN PROBABILITY MATRIX
    p_bar_train = createProbabilityMatrix(train_confrontation_matrix,
                                          estimator = "MAP", probability_model = ("uniform", None), 
                                          unknown_values = np.nan, diagonal_values = np.nan)
    
    # TEST PROBABILITY MATRIX
    p_bar_test = createProbabilityMatrix(test_confrontation_matrix, 
                                         estimator = "MAP", probability_model = ("uniform", None), 
                                         unknown_values = np.nan, diagonal_values = np.nan)

    
    for j in range(n_comparisons_completion):
        
        completion_method = to_compare_completion[j]
        print("METHOD ", completion_method[0])
        p_matrix, completion_time[j_tot][i] = complete(p_bar_train, train_selected_omega, train_confrontation_matrix,
                                             completion_method,
                                             in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
        
        _,_,_, accuracy[j_tot][i], _,_,_, _,_,_,_ = errorMetrics(p_matrix, p_bar_test, test_selected_omega, test_confrontation_matrix, completion_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
        
        j_tot += 1
        
    ######## MAXIMUM A POSTERIORI ON P TECHNIQUES ########
    
    # NO MATRICES TO COMPLETE 
    p_bar_train = None
    p_bar_test = None
    
    for j in range(n_comparisons_MAP_P):
        
        creation_method = to_compare_MAP_P[j]
        print("METHOD", creation_method[0])
        p_matrix, completion_time[j_tot][i] = create(train_confrontation_matrix, creation_method, 
                                                        estimator = "MAP", probability_model = ("uniform", None), 
                                                        in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
        
        _,_,_, accuracy[j_tot][i], _,_,_, _,_,_,_ = errorMetrics(p_matrix, p_bar_test, test_selected_omega, test_confrontation_matrix, completion_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
        
        j_tot += 1
        
    ######## MAXIMUM A POSTERIORI ON E TECHNIQUES ########    
    
    train_data, test_data, train_confrontation_tensor, test_confrontation_tensor, train_selected_omega, test_selected_omega, selected_players, selected_players_indices, selected_n_players, selected_extra_dimension_values, selected_extra_dimension_indices, selected_n_extra_dimension = selectingDataTensor(data, players, players_indices, extra_dimension_name, header, data_values, testing_percentage = testing_percentage, sorting_criterion = sorting_criterion, number_players = number_players, limit_dates = limit_dates)
    
    # NO MATRICES TO COMPLETE 
    p_bar_train = None
    p_bar_test = None
    
    for j in range(n_comparisons_MAP_E):
        creation_method = to_compare_MAP_E[j]
        print("METHOD ", creation_method[0])
        
        p_tensor, e_matrix, _, _, completion_time[j_tot][i] = createTensor(train_confrontation_tensor, creation_method, 
                                                              rating_model = rating_model, 
                                                              in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
        
        _,_,_, accuracy[j_tot][i], _,_,_,_,_,_,_ = errorMetricsTensor(p_tensor, e_matrix, p_bar_test, test_confrontation_tensor, creation_method, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half, clipping = clipping)
        
        j_tot += 1


plotComparisonSeasons(seasons, accuracy, to_compare)        

