#
# LINMA2990 - Master Thesis
# The mathematics of Tennis Ranking: Game outcome prediction by low-rank methods
#
# Data selection for 3D tensors
# 
# Bastien Massion
# Created on 14/04/2022
#
# Last version on 23/05/2022
#


import numpy as np
from Selection_thesis import *


# CONFRONTATION STATS

# Returns a matrix with the number of victories from player i on player j at entry (i,j)
# If no match has been played, stored as a 0
# Extra dimension can be Location, Tournament, Series, Court, Surface, Round, Score,...
def createConfrontationTensor(data, players, extra_dimension, header):
    name_extra_dimension = extra_dimension[0]
    values_extra_dimension = extra_dimension[1]
    n_values_extra_dimension = len(values_extra_dimension)
    n_players = len(players)
    confrontation_tensor = np.zeros((n_players, n_players, n_values_extra_dimension), dtype = int)
    
    header_winner = header.index('Winner')
    header_loser = header.index('Loser')
    header_extra_dimension = header.index(name_extra_dimension)
    n_data = len(data)
    for i in range(n_data):
        match = data[i]
        winner = match[header_winner]
        loser = match[header_loser]
        value_extra_dim = match[header_extra_dimension]
        winner_index = players.index(winner)
        loser_index = players.index(loser)
        value_extra_dim_index = values_extra_dimension.index(value_extra_dim)
        confrontation_tensor[winner_index,loser_index,value_extra_dim_index] += 1
    
    return confrontation_tensor

# omega contains all non-zero elements (only above the diagonal !)
def createOmegaTensor(confrontation_tensor):
    n_players = len(confrontation_tensor)
    n_extra_dim = len(confrontation_tensor[0,0,:])
    omega = []
    for i in range(n_players):
        for j in range(i, n_players):
            for k in range(n_extra_dim):
                if confrontation_tensor[i,j,k] != 0 or confrontation_tensor[j,i,k] != 0:
                    omega.append([i,j,k])
    return omega


def numberOfMatchesTensor(confrontation_tensor):
    n_players = len(confrontation_tensor)
    matches_wins_losses = np.zeros((3, n_players))
    matches_wins_losses[1,:] = np.sum(confrontation_tensor, axis = (1,2))
    matches_wins_losses[2,:] = np.sum(confrontation_tensor, axis = (0,2))
    matches_wins_losses[0,:] = matches_wins_losses[1,:] + matches_wins_losses[2,:]
    return matches_wins_losses 

# SORTING PLAYERS

def sortingPlayersTensor(players, players_indices, confrontation_tensor, sorting_criterion = "activity"):
    matches_wins_losses = numberOfMatchesTensor(confrontation_tensor)
    if sorting_criterion == "activity":
        sorted_players_indices = [player_index for _, player_index in sorted(zip(matches_wins_losses[0], players_indices), reverse = True)]
    elif sorting_criterion == "wins":
        sorted_players_indices = [player_index for _, player_index in sorted(zip(matches_wins_losses[1], players_indices), reverse = True)]
    elif sorting_criterion == "losses":
        sorted_players_indices = [player_index for _, player_index in sorted(zip(matches_wins_losses[2], players_indices), reverse = True)]
    elif sorting_criterion == "name":
        sorted_players_indices = players_indices.copy()
    else :
        print("Define a correct sorting criterion")
        return
    sorted_players = [players[i] for i in sorted_players_indices]
    return sorted_players, sorted_players_indices
   
def permutingPlayersTensor(tensor, sorted_indices):
    new_tensor_intermediate = tensor[sorted_indices, :, :]
    new_tensor = new_tensor_intermediate[:, sorted_indices, :]
    return new_tensor

def sortingExtraDimensionTensor(extra_dimension_values, extra_dimension_indices, train_confrontation_tensor):
    n_matches_extra_dimension = np.sum(train_confrontation_tensor, axis = (0,1))
    sorted_extra_dimension_indices = [player_index for _, player_index in sorted(zip(n_matches_extra_dimension, extra_dimension_indices), reverse = True)]
    sorted_extra_dimension_values = [extra_dimension_values[i] for i in sorted_extra_dimension_indices]
    return sorted_extra_dimension_values, sorted_extra_dimension_indices

def permutingExtraDimensionTensor(tensor, sorted_indices):
    new_tensor = tensor[:, :, sorted_indices]
    return new_tensor

# SELECTING PLAYERS

def selectingPlayersTensor(tensor, n_size):
    new_tensor = tensor[:n_size,:n_size,:]
    return new_tensor

def selectExtraDimensionAndIndices(sorted_extra_dimension_values, sorted_extra_dimension_indices, confrontation_tensor, lim = 5):
    n_matches_extra_dimension = np.sum(confrontation_tensor, axis = (0,1))
    n_max_extra_dimension = len(n_matches_extra_dimension)
    n_extra_dimension = n_max_extra_dimension
    for i in range(n_max_extra_dimension):
        if n_matches_extra_dimension[i] < lim:
            n_extra_dimension = i                                         
            break
        
    if n_extra_dimension > 0:
        return sorted_extra_dimension_values[:n_extra_dimension], sorted_extra_dimension_indices[:n_extra_dimension], n_extra_dimension
    else:
        print("Define a correct limiting criterion")
        return
    
def selectingExtraDimensionTensor(tensor, n_size):
    new_tensor = tensor[:,:,:n_size]
    return new_tensor


# ALL DATA SELECTION PROCESS
    
def selectingDataTensor(data, players, players_indices, extra_dimension_name, header, data_values, testing_percentage = 30.0, sorting_criterion = "activity", number_players = None, limit_dates = None):
    extra_dimension_values = data_values[header.index(extra_dimension_name)]
    n_extra_dimension = len(extra_dimension_values)
    extra_dimension = (extra_dimension_name, extra_dimension_values)
    
    if limit_dates != None :
        data = keepOnlyAllowedDates(data, header, limit_dates)
        
    # TRAINING AND TESTING SETS
    train_data, test_data = divideTrainingAndTestingData(data, testing_percentage = testing_percentage)
    
    # CONFRONTATION STATS
    train_confrontation_tensor = createConfrontationTensor(train_data, players, extra_dimension, header)
    test_confrontation_tensor = createConfrontationTensor(test_data, players, extra_dimension, header)
    
    # SORTING PLAYERS
    sorted_players, sorted_players_indices = sortingPlayersTensor(players, players_indices, train_confrontation_tensor, sorting_criterion = sorting_criterion)
    train_sorted_confrontation_tensor = permutingPlayersTensor(train_confrontation_tensor, sorted_players_indices)
    test_sorted_confrontation_tensor = permutingPlayersTensor(test_confrontation_tensor, sorted_players_indices)

    # SELECTING PLAYERS
    selected_players, selected_players_indices, selected_n_players = selectPlayersAndIndices(sorted_players, sorted_players_indices, n_players = number_players)
    train_selected_confrontation_tensor = selectingPlayersTensor(train_sorted_confrontation_tensor, selected_n_players)
    test_selected_confrontation_tensor = selectingPlayersTensor(test_sorted_confrontation_tensor, selected_n_players)
    
    # SORTING EXTRA DIMENSION
    extra_dimension_indices = np.arange(n_extra_dimension, dtype = int)
    sorted_extra_dimension_values, sorted_extra_dimension_indices = sortingExtraDimensionTensor(extra_dimension_values, extra_dimension_indices, train_selected_confrontation_tensor)
    train_sorted_confrontation_tensor = permutingExtraDimensionTensor(train_selected_confrontation_tensor, sorted_extra_dimension_indices)
    test_sorted_confrontation_tensor = permutingExtraDimensionTensor(test_selected_confrontation_tensor, sorted_extra_dimension_indices)
    
    # SELECTING EXTRA DIMENSION
    selected_extra_dimension_values, selected_extra_dimension_indices, selected_n_extra_dimension = selectExtraDimensionAndIndices(sorted_extra_dimension_values, sorted_extra_dimension_indices, train_sorted_confrontation_tensor, lim = 10)
    train_selected_confrontation_tensor = selectingExtraDimensionTensor(train_sorted_confrontation_tensor, selected_n_extra_dimension)
    test_selected_confrontation_tensor = selectingExtraDimensionTensor(test_sorted_confrontation_tensor, selected_n_extra_dimension)
    train_selected_omega = createOmegaTensor(train_selected_confrontation_tensor)
    test_selected_omega = createOmegaTensor(test_selected_confrontation_tensor)
    
    return train_data, test_data, train_selected_confrontation_tensor, test_selected_confrontation_tensor, train_selected_omega, test_selected_omega, selected_players, selected_players_indices, selected_n_players, selected_extra_dimension_values, selected_extra_dimension_indices, selected_n_extra_dimension
    
