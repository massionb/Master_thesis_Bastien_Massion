#
# LINMA2990 - Master Thesis
# The mathematics of Tennis Ranking: Game outcome prediction by low-rank methods
#
# Error metrics
# 
# Bastien Massion
# Created on 24/03/2022
#
# Last version on 14/04/2022
#


from Statistics_thesis import *
from Plotting_thesis import *
from Optimization_thesis import *

#Manipulate data
import numpy as np
import random

# ERROR MEASURE

def MSE(reconstructed_matrix, true_matrix, test_omega):
    n_known_elements = 2*len(test_omega)
    diff_projected = omegaProjector(true_matrix, test_omega) - omegaProjector(reconstructed_matrix, test_omega)
    mse = np.sum(diff_projected**2) / n_known_elements
    print("MSE : %.3f" %mse)
    return mse


def RMSE(reconstructed_matrix, true_matrix, test_omega):
    n_known_elements = 2*len(test_omega)
    diff_projected = omegaProjector(true_matrix, test_omega) - omegaProjector(reconstructed_matrix, test_omega)
    mse = np.sum(diff_projected**2) / n_known_elements
    rmse = np.sqrt(mse)
    print("RMSE : %.3f" %rmse)
    return rmse


def weigthedMSE(reconstructed_matrix, true_matrix, test_omega, test_confrontation_matrix):
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


def weigthedRMSE(reconstructed_matrix, true_matrix, test_omega, test_confrontation_matrix):
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


def matchesSimulations(reconstructed_matrix, test_selected_omega, test_confrontation_matrix, n_simulations = 100, epsilon = 10**(-6)):
    accuracy = np.zeros(n_simulations)
    for l in range(n_simulations):
        right_predictions = 0
        wrong_predictions = 0
        impossible_predictions = 0
        for k in range(len(test_selected_omega)):
            i = test_selected_omega[k][0]
            j = test_selected_omega[k][1]
            true_ij_wins = test_confrontation_matrix[i][j] 
            true_ji_wins = test_confrontation_matrix[j][i]
            ij_matches = true_ij_wins + true_ji_wins
            ij_wins_list = [1]*true_ij_wins 
            ji_wins_list = [0]*true_ji_wins
            true_outcome = [*ij_wins_list, *ji_wins_list]
            random.shuffle(true_outcome)
            predicted_prob_ij = reconstructed_matrix[i][j]
            predicted_prob_ji = reconstructed_matrix[j][i]
            if 0.0 <= predicted_prob_ij <= 1.0 and 0.0 <= predicted_prob_ji <= 1.0 and np.abs(predicted_prob_ij + predicted_prob_ji - 1.0) < epsilon :
                predicted_outcome = np.random.choice([1,0], size = (ij_matches,), p = [predicted_prob_ij, 1.0-predicted_prob_ij])
                for m in range(ij_matches):
                    if true_outcome[m] == predicted_outcome[m]:
                        right_predictions += 1
                    else:
                        wrong_predictions += 1
            else :
                impossible_predictions += ij_matches
        total_predictions = right_predictions + wrong_predictions + impossible_predictions
        accuracy[l] = right_predictions/total_predictions
    accuracy_mean = np.mean(accuracy)
    accuracy_std = np.std(accuracy)
    print("Accuracy of simulated (feasible) predictions : %.3f +- %.3f" %(accuracy_mean*100, accuracy_std*100))    
    return accuracy_mean, accuracy_std


def simulationsUpperBound(test_selected_omega, test_confrontation_matrix):
    good_predictions_clipping_equality = 0
    good_predictions_known = 0
    n_matches = 0
    for k in range(len(test_selected_omega)):
        i = test_selected_omega[k][0]
        j = test_selected_omega[k][1]
        true_ij_wins = test_confrontation_matrix[i][j] 
        true_ji_wins = test_confrontation_matrix[j][i]
        ij_matches = true_ij_wins + true_ji_wins
        wins_percentage_ij = true_ij_wins / ij_matches
        good_predictions_clipping_equality += ij_matches * (1/2 + np.abs(wins_percentage_ij - 1/2)) # Best case, when clipping(p_matrix) = clipping(p_bar_test MAP uniform)
        good_predictions_known += ij_matches * (1 - 2*wins_percentage_ij + 2*wins_percentage_ij**2) # If perfectly known, when p_matrix = p_bar_test MAP uniform
        n_matches += ij_matches
        
    upper_bound_accuracy_clipping_equality = good_predictions_clipping_equality / n_matches
    accuracy_known = good_predictions_known / n_matches
    print("Upper bound accuracy : %.3f" %upper_bound_accuracy_clipping_equality)
    print("Accuracy for perfect reconstruction : %.3f" %accuracy_known)
    return upper_bound_accuracy_clipping_equality, accuracy_known

def normalizedFrobenius(reconstructed_matrix, true_matrix, test_omega):
    diff_projected = omegaProjector(true_matrix, test_omega) - omegaProjector(reconstructed_matrix, test_omega)
    frob_diff = np.sqrt(np.sum(diff_projected**2))
    frob_reconstructed = np.sqrt(np.sum(omegaProjector(true_matrix, test_omega)**2))
    return frob_diff/frob_reconstructed


def rankMatrix(matrix):
    rank = np.linalg.matrix_rank(matrix)
    print("Rank of reconstructed matrix : %d" %rank)
    return rank


def singularValues(matrix):
    u, singular_values, v_t = np.linalg.svd(matrix)
    return u, singular_values, v_t


def respectOfConstrains(reconstructed_matrix, in_range = True, add_to_one = True, diagonal_half = True, epsilon = 10**(-6)):
    out_of_range = 0
    not_adding_to_one = 0
    not_half_on_diagonal = 0
    n_players = len(reconstructed_matrix)
    for i in range(n_players):
        for j in range(n_players):
            if reconstructed_matrix[i][j] < 0.0 or reconstructed_matrix[i][j] > 1.0:
                out_of_range += 1
            if i < j and np.abs(reconstructed_matrix[i][j] + reconstructed_matrix[j][i] - 1.0) > epsilon:
                not_adding_to_one += 1   
            if i == j and np.abs(reconstructed_matrix[i][j] - 0.5) > epsilon:
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
        print("Number of diagonal entries != 0.5 : 0 (imposed)")    
    else :
        print("Number of diagonal entries != 0.5 :", not_half_on_diagonal)
    return out_of_range, not_adding_to_one, not_half_on_diagonal


def errorMetrics(p_matrix, p_bar_test, test_selected_omega, test_confrontation_matrix, completion_method, in_range = True, add_to_one = True, diagonal_half = True, clipping = True):
    if p_bar_test is not None :
        rmse = RMSE(p_matrix, p_bar_test, test_selected_omega)
        wrmse = weigthedRMSE(p_matrix, p_bar_test, test_selected_omega, test_confrontation_matrix)
    else :
        rmse = None
        print("RMSE : no matrix to compare to")
        wrmse = None
        print("WRMSE : no matrix to compare to")
    u, singular_values, v_t = singularValues(p_matrix)
    u_centered, singular_values_centered, v_t_centered = singularValues(p_matrix - 0.5*np.ones(np.shape(p_matrix)))
#    plotSingularValues(singular_values, singular_values_centered, completion_method, clipping)
    p_matrix_respect = makeRespectConstraints(p_matrix)
    acc_mean, acc_std = matchesSimulations(p_matrix_respect, test_selected_omega, test_confrontation_matrix)
    acc_upper_bound, acc_known = simulationsUpperBound(test_selected_omega, test_confrontation_matrix)
    out_of_range, not_adding_to_one, not_half_on_diagonal = respectOfConstrains(p_matrix, in_range = in_range, add_to_one = add_to_one, diagonal_half = diagonal_half)
    rank = rankMatrix(p_matrix)
    return rmse, wrmse, singular_values, acc_mean, acc_std, acc_upper_bound, acc_known, out_of_range, not_adding_to_one, not_half_on_diagonal, rank

def makeRespectConstraints(p_matrix):
    p_matrix_respect = np.copy(p_matrix)
    p_matrix_respect = inRange(p_matrix_respect)
    p_matrix_respect = addToOne(p_matrix_respect)
    p_matrix_respect = halfOnDiagonal(p_matrix_respect)
    return p_matrix_respect
    

def createMetricsMatrices(n_comparisons, n_players):
    rmse = np.zeros((n_comparisons,2))
    wrmse = np.zeros((n_comparisons,2))
    singular_values = np.zeros((n_comparisons,2,n_players))
    acc_mean = np.zeros((n_comparisons,2))
    acc_std = np.zeros((n_comparisons,2))
    acc_upper_bound = 0
    acc_known = 0
    out_of_range = np.zeros((n_comparisons,2))
    not_adding_to_one = np.zeros((n_comparisons,2))
    not_half_on_diagonal = np.zeros((n_comparisons,2))
    rank = np.zeros((n_comparisons,2))
    return rmse, wrmse, singular_values, acc_mean, acc_std, acc_upper_bound, acc_known, out_of_range, not_adding_to_one, not_half_on_diagonal, rank