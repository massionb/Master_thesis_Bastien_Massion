#
# LINMA2990 - Master Thesis
# The mathematics of Tennis Ranking: Game outcome prediction by low-rank methods
#
# Error metrics
# 
# Bastien Massion
# Created on 17/04/2022
#
# Last version on 17/04/2022
#


from Statistics_thesis import *
from Plotting_thesis import *

#Manipulate data
import numpy as np
import random

# ERROR MEASURE
#not adapted yet
def MSETensor(reconstructed_matrix, true_matrix, test_omega):
    n_known_elements = 2*len(test_omega)
    diff_projected = omegaProjector(true_matrix, test_omega) - omegaProjector(reconstructed_matrix, test_omega)
    mse = np.sum(diff_projected**2) / n_known_elements
    print("MSE : %.3f" %mse)
    return mse

#not adapted yet
def RMSETensor(reconstructed_matrix, true_matrix, test_omega):
    n_known_elements = 2*len(test_omega)
    diff_projected = omegaProjector(true_matrix, test_omega) - omegaProjector(reconstructed_matrix, test_omega)
    mse = np.sum(diff_projected**2) / n_known_elements
    rmse = np.sqrt(mse)
    print("RMSE : %.3f" %rmse)
    return rmse

#not adapted yet
def weigthedMSETensor(reconstructed_matrix, true_matrix, test_omega, test_confrontation_matrix):
    diff_projected = omegaProjector(true_matrix, test_omega) - omegaProjector(reconstructed_matrix, test_omega)
    n_players = len(diff_projected[:,0])
    weights_matrix = np.zeros((n_players,n_players))
    for k in range(len(test_omega)):
        i = test_omega[k][0]
        j = test_omega[k][1]
        weights_matrix[i][j] = test_confrontation_matrix[i][j] + test_confrontation_matrix[j][i]
        weights_matrix[j][i] = test_confrontation_matrix[i][j] + test_confrontation_matrix[j][i]

    n_total_matches = np.sum(weights_matrix) / 2
    wmse = np.sum(weights_matrix * diff_projected**2) / (2*n_total_matches)
    print("Weighted MSE : %.3f" %wmse)
    return wmse

#not adapted yet
def weigthedRMSETensor(reconstructed_matrix, true_matrix, test_omega, test_confrontation_matrix):
    diff_projected = omegaProjector(true_matrix, test_omega) - omegaProjector(reconstructed_matrix, test_omega)
    n_players = len(diff_projected[:,0])
    weights_matrix = np.zeros((n_players,n_players))
    for k in range(len(test_omega)):
        i = test_omega[k][0]
        j = test_omega[k][1]
        weights_matrix[i][j] = test_confrontation_matrix[i][j] + test_confrontation_matrix[j][i]
        weights_matrix[j][i] = test_confrontation_matrix[i][j] + test_confrontation_matrix[j][i]
        
    n_total_matches = np.sum(weights_matrix) / 2
    wmse = np.sum(weights_matrix * diff_projected**2) / (2*n_total_matches)
    wrmse = np.sqrt(wmse)
    print("Weighted RMSE : %.3f" %wrmse)
    return wrmse

def matchesSimulationsTensor(reconstructed_tensor, test_confrontation_tensor, n_simulations = 100, epsilon = 10**(-6)):
    accuracy = np.zeros(n_simulations)
    n_players = len(test_confrontation_tensor)
    n_extra_dimension = len(test_confrontation_tensor[0,0,:])
    for l in range(n_simulations):
        right_predictions = 0
        wrong_predictions = 0
        impossible_predictions = 0
        for i in range(n_players):
            for j in range(n_players):
                for c in range(n_extra_dimension):
                    true_ijc_wins = test_confrontation_tensor[i][j][c]
                    true_jic_wins = test_confrontation_tensor[j][i][c]
                    ijc_matches = true_ijc_wins + true_jic_wins
                    if ijc_matches != 0:
                        ijc_wins_list = [1]*true_ijc_wins 
                        jic_wins_list = [0]*true_jic_wins
                        true_outcome = [*ijc_wins_list, *jic_wins_list]
                        random.shuffle(true_outcome)
                        predicted_prob_ijc = reconstructed_tensor[i][j][c]
                        predicted_prob_jic = reconstructed_tensor[j][i][c]
                        if 0.0 <= predicted_prob_ijc <= 1.0 and 0.0 <= predicted_prob_jic <= 1.0 and np.abs(predicted_prob_ijc + predicted_prob_jic - 1.0) < epsilon :
                            predicted_outcome = np.random.choice([1,0], size = (ijc_matches,), p = [predicted_prob_ijc, 1.0-predicted_prob_ijc])
                            for m in range(ijc_matches):
                                if true_outcome[m] == predicted_outcome[m]:
                                    right_predictions += 1
                                else:
                                    wrong_predictions += 1
                        else :
                            impossible_predictions += ijc_matches
        total_predictions = right_predictions + wrong_predictions + impossible_predictions
        accuracy[l] = right_predictions/total_predictions
    accuracy_mean = np.mean(accuracy)
    accuracy_std = np.std(accuracy)
    print("Accuracy of simulated (feasible) predictions : %.3f +- %.3f" %(accuracy_mean*100, accuracy_std*100))    
    return accuracy_mean, accuracy_std


def simulationsUpperBoundTensor(test_confrontation_tensor):
    n_players = len(test_confrontation_tensor)
    n_extra_dimension = len(test_confrontation_tensor[0,0,:])
    good_predictions_clipping_equality = 0
    good_predictions_known = 0
    n_matches = 0
    for i in range(n_players):
        for j in range(n_players):
            for c in range(n_extra_dimension):
                true_ijc_wins = test_confrontation_tensor[i][j][c]
                true_jic_wins = test_confrontation_tensor[j][i][c]
                ijc_matches = true_ijc_wins + true_jic_wins
                if ijc_matches != 0:
                    wins_percentage_ijc = true_ijc_wins / ijc_matches
                    good_predictions_clipping_equality += ijc_matches * (1/2 + np.abs(wins_percentage_ijc - 1/2)) # Best case, when clipping(p_matrix) = clipping(p_bar_test MAP uniform)
                    good_predictions_known += ijc_matches * (1 - 2*wins_percentage_ijc + 2*wins_percentage_ijc**2) # If perfectly known, when p_matrix = p_bar_test MAP uniform
                    n_matches += ijc_matches
        
    upper_bound_accuracy_clipping_equality = good_predictions_clipping_equality / n_matches
    accuracy_known = good_predictions_known / n_matches
    print("Upper bound accuracy : %.3f" %upper_bound_accuracy_clipping_equality)
    print("Accuracy for perfect reconstruction : %.3f" %accuracy_known)
    return upper_bound_accuracy_clipping_equality, accuracy_known


def rankMatrix(matrix):
    rank = np.linalg.matrix_rank(matrix)
    print("Rank of reconstructed matrix : %d" %rank)
    return rank


def singularValues(matrix):
    u, singular_values, v_t = np.linalg.svd(matrix)
    return u, singular_values, v_t


def respectOfConstrains(reconstructed_tensor, in_range = True, add_to_one = True, diagonal_half = True, epsilon = 10**(-6)):
    out_of_range = 0
    not_adding_to_one = 0
    not_half_on_diagonal = 0
    n_players = len(reconstructed_tensor)
    n_extra_dimension = len(reconstructed_tensor[0,0,:])
    for i in range(n_players):
        for j in range(n_players):
            for c in range(n_extra_dimension):
                if reconstructed_tensor[i][j][c] < 0.0 or reconstructed_tensor[i][j][c] > 1.0:
                    out_of_range += 1
                if i < j and np.abs(reconstructed_tensor[i][j][c] + reconstructed_tensor[j][i][c] - 1.0) > epsilon:
                    not_adding_to_one += 1   
                if i == j and np.abs(reconstructed_tensor[i][j][c] - 0.5) > epsilon:
                    not_half_on_diagonal += 1
    if in_range == True:
        print("Number of out of range elements (p_ij < 0.0 or p_ij > 1.0) : 0 (imposed)")
    else:
        print("Number of out of range elements (p_ij < 0.0 or p_ij > 1.0) :", out_of_range)
    if add_to_one == True:
        print("Number of  pairs such that p_ij + p_ji != 1.0 : 0 (imposed)")
    else:
        print("Number of pairs such that p_ij + p_ji != 1.0 :", not_adding_to_one)
    if diagonal_half == True:
        print("Number of diagonal entries not equal to 0.5 : 0 (imposed)")    
    else :
        print("Number of diagonal entries not equal to 0.5 :", not_half_on_diagonal)
    return out_of_range, not_adding_to_one, not_half_on_diagonal


def errorMetricsTensor(p_tensor, e_matrix, p_bar_test, test_confrontation_tensor, completion_method, in_range = True, add_to_one = True, diagonal_half = True, clipping = True):
    if p_bar_test is not None :
        print("NOT SUPPOSED TO COME HERE")
        rmse = RMSETensor(p_matrix, p_bar_test, test_selected_omega)
        wrmse = weigthedRMSETensor(p_matrix, p_bar_test, test_selected_omega, test_confrontation_matrix)
    else :
        rmse = None
        print("RMSE : no matrix to compare to")
        wrmse = None
        print("WRMSE : no matrix to compare to")
    u, singular_values, v_t = singularValues(e_matrix)
#    u_centered, singular_values_centered, v_t_centered = singularValues(e_matrix - 0.5*np.ones(np.shape(e_matrix)))
#    plotSingularValues(singular_values, singular_values_centered, completion_method)
    acc_mean, acc_std = matchesSimulationsTensor(p_tensor, test_confrontation_tensor)
    acc_upper_bound, acc_known = simulationsUpperBoundTensor(test_confrontation_tensor)
    out_of_range, not_adding_to_one, not_half_on_diagonal = respectOfConstrains(p_tensor, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half)
    rank = rankMatrix(e_matrix)
    return rmse, wrmse, singular_values, acc_mean, acc_std, acc_upper_bound, acc_known, out_of_range, not_adding_to_one, not_half_on_diagonal, rank


def createMetricsMatricesTensor(n_comparisons, n_players, n_extra_dimension):
    rmse = np.zeros((n_comparisons,2))
    wrmse = np.zeros((n_comparisons,2))
    singular_values = np.zeros((n_comparisons,2,min(n_players,n_extra_dimension)))
    acc_mean = np.zeros((n_comparisons,2))
    acc_std = np.zeros((n_comparisons,2))
    acc_upper_bound = 0
    acc_known = 0
    out_of_range = np.zeros((n_comparisons,2))
    not_adding_to_one = np.zeros((n_comparisons,2))
    not_half_on_diagonal = np.zeros((n_comparisons,2))
    rank = np.zeros((n_comparisons,2))
    return rmse, wrmse, singular_values, acc_mean, acc_std, acc_upper_bound, acc_known, out_of_range, not_adding_to_one, not_half_on_diagonal, rank