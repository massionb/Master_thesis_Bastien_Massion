#
# LINMA2990 - Master Thesis
# The mathematics of Tennis Ranking: Game outcome prediction by low-rank methods
#
# Statistics about players
# 
# Bastien Massion
# Created on 03/01/2022
#
# Last version on 29/03/2022
#

#Manipulate data
import numpy as np
from scipy.special import comb
from scipy.integrate import quad
from scipy.optimize import minimize, Bounds

  
# PROBLEM INFORMATION
    
def percentageKnownElements(omega, n_players, diagonal_values = 0.5, diag_into_account = False):
    n_known_elements = 2*len(omega)
    n_elements = n_players**2 - n_players
    if diag_into_account == True:
        if np.isnan(diagonal_values) == False:
            n_known_elements += n_players
        n_elements += n_players
    n_unknown_elements = n_elements - n_known_elements
    percentage_known_elements = n_known_elements/n_elements   # don't count for the diagonal
    print("Number of players : %d" %n_players)
    if diag_into_account == True:
        print("Number of elements : %d (diagonal included)" %n_unknown_elements)
        print("Number of unknown elements : %d (diagonal included)" %n_unknown_elements)
        print("Percentage of known elements : %.1f (diagonal included)" %(percentage_known_elements*100))
    else:
        print("Number of elements : %d" %n_elements)
        print("Number of unknown pairs : %d" %(n_unknown_elements/2))
        print("Percentage of known elements : %.1f" %(percentage_known_elements*100))
    return percentage_known_elements, n_unknown_elements
    
    
# PROBABILITY MATRIX

# Lots of elements missing (variables to predict by matrix completion) stored as -1
def createProbabilityMatrix(confrontation_matrix, estimator = "MAP", probability_model = ("uniform", None), unknown_values = np.nan, diagonal_values = np.nan):
    model_name = probability_model[0]
    model_parameters = probability_model[1]
    n_players = len(confrontation_matrix)
    prob_matrix = - np.ones((n_players, n_players)) # initialize matrix as -1
    for i in range(n_players):
        for j in range(n_players):
            ij_wins = confrontation_matrix[i][j]
            ji_wins = confrontation_matrix[j][i]
            
            if ij_wins > 0 or ji_wins > 0:
                if estimator == "MAP":
                    p_ij_bar = MAPEstimator(model_name, model_parameters, ij_wins, ji_wins)
                elif estimator == "conditional_mean":
                    p_ij_bar = conditionalMeanEstimator(model_name, model_parameters, ij_wins, ji_wins)
                else:
                    print("Define a correct estimator")
                    return
                prob_matrix[i][j] = p_ij_bar
                prob_matrix[j][i] = 1.0 - prob_matrix[i][j]

    for i in range(n_players):
        for j in range(n_players):
            if i == j:                
                prob_matrix[i][j] = diagonal_values
            elif prob_matrix[i][j] == -1:
                prob_matrix[i][j] = unknown_values
    return prob_matrix


# PROBABILITY DISTRIBUTIONS

def binomialDistribution(p_ij, w_ij, m_ij):
    if 0.0 <= p_ij <= 1.0:
        return  comb(m_ij, w_ij, exact = True) * p_ij**(w_ij) * (1.0 - p_ij)**(m_ij-w_ij)
    else :
        print("p_ij not in [0,1]")
        return

def uniformDistribution(p_ij):
    if 0.0 <= p_ij <= 1.0:
        return 1.0
    else :
        print("p_ij not in [0,1]")
        return
    
def exponentialLikeDistribution(p_ij, parameter_a): # Supposing parameter_a > 1
    if p_ij == 0.0 or p_ij == 1.0 :
        return 0.0    
    elif 0.0 < p_ij <= 0.5 :
        return parameter_a/2.0 * p_ij**(parameter_a - 1.0) * (1.0-p_ij)**(-parameter_a - 1.0)
    elif 0.5 <= p_ij < 1.0:
        return parameter_a/2.0 * p_ij**(-parameter_a - 1.0) * (1.0-p_ij)**(parameter_a - 1.0)
    else :
        print("p_ij not in [0,1]")
        return

def logisticLikeDistribution(p_ij, parameter_a): # Supposing parameter_a > 1
    if p_ij == 0.0 or p_ij == 1.0:
        return 0.0
    elif 0.0 < p_ij < 0.5 or 0.5 < p_ij < 1.0:
        frac = p_ij/(1.0-p_ij)
        num = parameter_a * frac**(parameter_a) * (-2*frac**(parameter_a) + parameter_a * frac**(parameter_a) * np.log(frac) + parameter_a * np.log(frac) + 2)
        den = p_ij * (1.0-p_ij) * (frac**(parameter_a) - 1.0)**3
        return num/den
    elif p_ij == 0.5:
        return 2.0*parameter_a/3.0
    else :
        print("p_ij not in [0,1]")
        return

def posteriorNonNormalizedDistribution(p_ij, ij_wins, ji_wins, model, model_parameters):
    if ij_wins != 0 or ji_wins != 0:
        ij_total_matches = ij_wins + ji_wins
        if model == "uniform":
            return binomialDistribution(p_ij, ij_wins, ij_total_matches)*uniformDistribution(p_ij)
        elif model == "exponential_like":
            return binomialDistribution(p_ij, ij_wins, ij_total_matches)*exponentialLikeDistribution(p_ij, model_parameters)
        elif model == "logistic_like":
            return binomialDistribution(p_ij, ij_wins, ij_total_matches)*logisticLikeDistribution(p_ij, model_parameters)
        else:
            print("Define a correct probability model")
            return
    else:
        print("No matches played : should not care about this distribution")
        return

def posteriorNormalizedDistribution(p_ij, ij_wins, ji_wins, model, model_parameters):
    if ij_wins != 0 or ji_wins != 0:
        normalization_factor = quad(posteriorNonNormalizedDistribution, 0.0, 1.0, args = (ij_wins, ji_wins, model, model_parameters))[0]
        return posteriorNonNormalizedDistribution(p_ij, ij_wins, ji_wins, model, model_parameters)/normalization_factor
    else:
        print("No matches played : should not care about this distribution")
        return
    

def uniformMAPEstimator(ij_wins, ji_wins): # Max-Likelihood (ML) estimator
    if ij_wins != 0 or ji_wins != 0:
        ij_total_matches = ij_wins + ji_wins
        p_ij_bar = ij_wins/ij_total_matches
        return p_ij_bar
    else:
        print("No matches played : should not care about this estimator")
        return

def betaMAPEstimator(b, ij_wins, ji_wins):
    if ij_wins != 0 or ji_wins != 0:
        ij_total_matches = ij_wins + ji_wins
        p_ij_bar = (ij_wins+b-1)/(ij_total_matches +2*b-2)
        return p_ij_bar
    else:
        print("No matches played : should not care about this estimator")
        return

def exponentialLikeMAPEstimator(parameter_a, ij_wins, ji_wins): # Max-A-Posteriori (MAP) estimator with exponential-like prior
    if ij_wins != 0 or ji_wins != 0:
        ij_total_matches = ij_wins + ji_wins
        if ij_total_matches <= 2:
            p_ij_bar = 0.5
        else:
            if 0 <= ij_wins <= ij_total_matches/2.0 - parameter_a :
                p_ij_bar = (ij_wins + parameter_a - 1.0) / (ij_total_matches - 2.0)
            elif ij_total_matches/2.0 - parameter_a <= ij_wins <= ij_total_matches/2.0 + parameter_a :
                p_ij_bar = 0.5
            else :
                p_ij_bar = (ij_wins - parameter_a - 1.0) / (ij_total_matches - 2.0)
        return p_ij_bar
    else:
        print("No matches played : should not care about this estimator")
        return

def logisticLikeMAPEstimator(parameter_a, ij_wins, ji_wins, eps = 10**(-5)):
    if ij_wins != 0 or ji_wins != 0:
        minimum = minimize(logPosteriorDistribution, 0.5, args = (ij_wins, ji_wins, "logistic_like", parameter_a), method = 'Powell', bounds = [(0.0+eps, 1.0-eps)])
        p_ij_bar = minimum.get('x')[0]
        return p_ij_bar
    else:
        print("No matches played : should not care about this estimator")
        return


def MAPEstimator(model, model_parameters, ij_wins, ji_wins):
    if model == "uniform":
        p_ij_bar = uniformMAPEstimator(ij_wins, ji_wins)
    elif model == "exponential_like":
        p_ij_bar = exponentialLikeMAPEstimator(model_parameters, ij_wins, ji_wins)
    elif model == "logistic_like":
        p_ij_bar = logisticLikeMAPEstimator(model_parameters, ij_wins, ji_wins)
    elif model == "beta":
        p_ij_bar = betaMAPEstimator(model_parameters, ij_wins, ji_wins)
    else :
        print("Define a correct probability model")
        return
    return p_ij_bar
    
    
def posteriorNormalizedTimesPijDistribution(p_ij, ij_wins, ji_wins, model, model_parameters):
    return p_ij*posteriorNormalizedDistribution(p_ij, ij_wins, ji_wins, model, model_parameters)
    
def conditionalMeanEstimator(model, model_parameters, ij_wins, ji_wins): # Also Minimum MSE estimator
#    p_ij_bar = quad(posteriorNormalizedTimesPijDistribution, 0.0, 1.0, args = (ij_wins, ji_wins, model, model_parameters))[0]
    if model == "uniform":
        p_ij_bar = uniformCMEstimator(ij_wins, ji_wins)
    elif model == "exponential_like":
        p_ij_bar = exponentialLikeCMEstimator(model_parameters, ij_wins, ji_wins)
    elif model == "logistic_like":
        p_ij_bar = logisticLikeCMEstimator(model_parameters, ij_wins, ji_wins)
    elif model == "beta":
        p_ij_bar = betaCMEstimator(model_parameters, ij_wins, ji_wins)
    else :
        print("Define a correct probability model")
        return
    return p_ij_bar

def uniformCMEstimator(ij_wins, ji_wins):
    if ij_wins != 0 or ji_wins != 0:
        ij_total_matches = ij_wins + ji_wins
        p_ij_bar = (ij_wins+1)/(ij_total_matches+2)
        return p_ij_bar
    else:
        print("No matches played : should not care about this estimator")
        return
    
def betaCMEstimator(b, ij_wins, ji_wins):
    if ij_wins != 0 or ji_wins != 0:
        ij_total_matches = ij_wins + ji_wins
        p_ij_bar = (ij_wins+b)/(ij_total_matches+2*b)
        return p_ij_bar
    else:
        print("No matches played : should not care about this estimator")
        return

# OBJECTIVE FUNCTIONS

def logPosteriorDistribution(p_ij, ij_wins, ji_wins, model, model_parameters): # without normalization
    if p_ij <= 0.0 or p_ij >= 1.0 :
        return np.infty
    else:
        return - np.log(posteriorNonNormalizedDistribution(p_ij, ij_wins, ji_wins, model, model_parameters))


def omegaProjector(matrix, omega, binary_mask = False):
    n_players = len(matrix)
    projected_matrix = np.zeros((n_players,n_players))
    mask = np.zeros((n_players,n_players), dtype = np.int)
    for k in range(len(omega)):
        i = omega[k][0]
        j = omega[k][1]
        projected_matrix[i][j] = matrix[i][j] # Above diagonal 
        projected_matrix[j][i] = matrix[j][i] # Under diagonal 
        mask[i][j] = 1
        mask[j][i] = 1
    if binary_mask == False:
        return projected_matrix
    else :
        return projected_matrix, mask
        
