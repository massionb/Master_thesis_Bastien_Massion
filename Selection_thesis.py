#
# LINMA2990 - Master Thesis
# The mathematics of Tennis Ranking: Game outcome prediction by low-rank methods
#
# Data selection
# 
# Bastien Massion
# Created on 29/03/2022
#
# Last version on 23/05/2022
#


import numpy as np
from datetime import datetime

# SELECTING DATES

def keepOnlyAllowedDates(data, header, limit_dates):
    start_date = datetime.strptime(limit_dates[0], '%d/%m/%Y')
    end_date = datetime.strptime(limit_dates[1], '%d/%m/%Y')
    header_date = header.index('Date')
    
    n_data = len(data)
    dates_to_keep = []
    for i in range(n_data):
        match = data[i]
        date = match[header_date]
        if start_date <= date <= end_date:
            dates_to_keep.append(i)
    
    data = [data[i] for i in dates_to_keep]
    
    return data


# TRAINING AND TESTING SETS

def divideTrainingAndTestingData(data, testing_percentage = 30.0):
    n_data = len(data)
    limit_index = int(n_data*(100.0-testing_percentage)/100.0)
    arr = np.arange(n_data)
    np.random.shuffle(arr)
    data_indices = list(arr)
    train_data_indices = sorted(data_indices[:limit_index])
    train_data = []
    for i in train_data_indices :
        train_data.append(data[i])
    test_data_indices = sorted(data_indices[limit_index:])
    test_data = []
    for i in test_data_indices :
        test_data.append(data[i])
    return train_data, test_data


# CONFRONTATION STATS

# Returns a matrix with the number of victories from player i on player j at entry (i,j)
# If no match has been played, stored as a 0
def createConfrontationMatrix(data, players, header):
    n_players = len(players)
    confrontation_matrix = np.zeros((n_players, n_players), dtype = int)
    
    header_winner = header.index('Winner')
    header_loser = header.index('Loser')
    n_data = len(data)
    for i in range(n_data):
        match = data[i]
        winner = match[header_winner]
        loser = match[header_loser]
        winner_index = players.index(winner)
        loser_index = players.index(loser)
        confrontation_matrix[winner_index][loser_index] += 1
    
    return confrontation_matrix

# omega contains all non-zero elements (only above the diagonal !)
def createOmega(confrontation_matrix):
    n_players = len(confrontation_matrix)
    omega = []
    for i in range(n_players):
        for j in range(i, n_players):
            if confrontation_matrix[i][j] != 0 or confrontation_matrix[j][i] != 0:
                omega.append([i,j])
    return omega

def numberOfMatches(confrontation_matrix, omega):
    n_players = len(confrontation_matrix)
    n_matches = []
    n_wins = []
    n_loss = []
    for i in range(n_players):
        n_wins.append(0)
        n_loss.append(0)
        n_matches.append(0)
    for ij in omega : 
        i = ij[0]
        j = ij[1]
        if confrontation_matrix[i][j] != 0:
            n_wins[i] += confrontation_matrix[i][j]
            n_matches[i] += confrontation_matrix[i][j]
            n_loss[j] += confrontation_matrix[i][j]
            n_matches[j] += confrontation_matrix[i][j]
        if confrontation_matrix[j][i] != 0:
            n_loss[i] += confrontation_matrix[j][i]
            n_matches[i] += confrontation_matrix[j][i]
            n_wins[j] += confrontation_matrix[j][i]
            n_matches[j] += confrontation_matrix[j][i]
    return [n_matches, n_wins, n_loss]


# SORTING PLAYERS

def sorting(players, players_indices, confrontation_matrix, sorting_criterion = "activity"):
    omega = createOmega(confrontation_matrix)
    matches_wins_losses = numberOfMatches(confrontation_matrix, omega)
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
   
def permutingMatrix(matrix, sorted_indices):
    new_matrix_intermediate = matrix[sorted_indices, :]
    new_matrix = new_matrix_intermediate[:, sorted_indices]
    return new_matrix


# SELECTING PLAYERS

def selectPlayersAndIndices(sorted_players, sorted_players_indices, n_players = None):
    if n_players != None:
        selected_n_players = min(n_players,len(sorted_players))
        return sorted_players[:selected_n_players], sorted_players_indices[:selected_n_players], selected_n_players
    else :
        print("Define a correct selection criterion")
        return

def selectingMatrix(matrix, n_size):
    new_matrix = matrix[:n_size,:n_size]
    return new_matrix

# ALL DATA SELECTION PROCESS
    
def selectingData(data, players, players_indices, header, testing_percentage = 30.0, sorting_criterion = "activity", number_players = None, limit_dates = None):
    if limit_dates != None :
        data = keepOnlyAllowedDates(data, header, limit_dates)
        
    # TRAINING AND TESTING SETS
    train_data, test_data = divideTrainingAndTestingData(data, testing_percentage = testing_percentage)
    
    # CONFRONTATION STATS
    train_confrontation_matrix = createConfrontationMatrix(train_data, players, header)
    test_confrontation_matrix = createConfrontationMatrix(test_data, players, header)
    
    # SORTING PLAYERS
    sorted_players, sorted_players_indices = sorting(players, players_indices, train_confrontation_matrix, sorting_criterion = sorting_criterion)
    train_sorted_confrontation_matrix = permutingMatrix(train_confrontation_matrix, sorted_players_indices)
    test_sorted_confrontation_matrix = permutingMatrix(test_confrontation_matrix, sorted_players_indices)
    
    # SELECTING PLAYERS
    selected_players, selected_players_indices, selected_n_players = selectPlayersAndIndices(sorted_players, sorted_players_indices, n_players = number_players)
    train_selected_confrontation_matrix = selectingMatrix(train_sorted_confrontation_matrix, selected_n_players)
    test_selected_confrontation_matrix = selectingMatrix(test_sorted_confrontation_matrix, selected_n_players)
    train_selected_omega = createOmega(train_selected_confrontation_matrix)
    test_selected_omega = createOmega(test_selected_confrontation_matrix)
    
    return train_data, test_data, train_selected_confrontation_matrix, test_selected_confrontation_matrix, train_selected_omega, test_selected_omega, selected_players, selected_players_indices, selected_n_players
    
