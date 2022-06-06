#
# LINMA2990 - Master Thesis
# The mathematics of Tennis Ranking: Game outcome prediction by low-rank methods
#
# Plotting
# 
# Bastien Massion
# Created on 01/03/2022
#
# Last version on 14/04/2022
#


#Plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import savgol_filter

#Distributions
from Prior_thesis import *
from Statistics_thesis import *


# PLOTTING DISTRIBUTIONS

def plotDistributions(ij_wins, ji_wins, models, models_parameters):
    n_models = len(models)
    n_points = 1000
    x_points = np.linspace(0.0,1.0,n_points)
    y_points = np.zeros((n_models,n_points))
    y_points_bis = np.zeros((n_models,n_points))
    y_max = 0.0
    for k in range(n_models):
        for l in range(n_points):
            y_points[k,l] = posteriorNormalizedDistribution(x_points[l], ij_wins, ji_wins, models[k], models_parameters[k])
            y_points_bis[k,l] = logPosteriorDistribution(x_points[l], ij_wins, ji_wins, models[k], models_parameters[k])
        y_max = max(y_max, np.max(y_points))
        
    plt.figure("Distributions", figsize = (20,15)) 
    plt.title("$w_{ij} =$ %d, $m_{ij} = $ %d" %(ij_wins, ij_wins+ji_wins), fontsize=30)
    plt.grid() 
    for k in range(n_models):
        plt.plot(x_points, y_points[k], label = models[k] + ", param = " + str(models_parameters[k]))
    plt.vlines(0.5, 0.0 - y_max*0.07, y_max*1.15, color = 'darkgrey', linestyles = 'dashed')
    plt.legend(fontsize=30) #Ajouter une légende à la l'image
    plt.ylim((0.0 - y_max*0.1,y_max*1.2))
    plt.xlabel("$p_{ij}$", fontsize=30) #Label pour l'axe des x
    plt.ylabel("PDF", fontsize=30) #Label pour l'axe des x
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30) #Taille des valeurs des axes
    plt.savefig('../Figures/Distributions.png') #Sauvergarder l'image
    plt.show() #Montrer l'image
    
    plt.figure("Log-distributions", figsize = (20,15)) 
    plt.title("$w_{ij} =$ %d, $m_{ij} = $ %d" %(ij_wins, ij_wins+ji_wins), fontsize=30)
    plt.grid() 
    for k in range(len(models)):
        plt.plot(x_points, y_points_bis[k], label = "log-" + models[k] + ", param = " + str(models_parameters[k]))
    plt.vlines(0.5, 0.0 - y_max*0.07, y_max*1.15, color = 'darkgrey', linestyles = 'dashed')
    plt.legend(fontsize=30) #Ajouter une légende à la l'image
    plt.ylim((0.0 - y_max*0.1,y_max*1.2))
    plt.xlabel("$log(p_{ij})$", fontsize=30) #Label pour l'axe des x
    plt.ylabel("PDF", fontsize=30) #Label pour l'axe des x
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30) #Taille des valeurs des axes
    plt.savefig('../Figures/LogDistributions.png') #Sauvergarder l'image
    plt.show() #Montrer l'image
    

def plotGamesRanking(players_rankings):
    plt.figure("Wins and rankings", figsize = (20,15)) 
    plt.title("Difference of points for winner", fontsize=30)
    plt.grid()
    labels = ["winners", "losers", "difference"]
    for line in range(3): #0 = winners, 1 = losers, 2 = difference
        label_i = labels[line]
        min_rank = int(np.nanmin(players_rankings[line]))
        max_rank = int(np.nanmax(players_rankings[line]))
        x_points = np.arange(min_rank, max_rank)
        y_points = np.zeros(len(x_points))
        y_nan = 0
        for ranking in players_rankings[line]:
            if np.isnan(ranking) == False:
                ranking = int(ranking)
                y_points[np.where(x_points == ranking)] += 1
            else :
                y_nan += 1
        plt.plot(x_points, y_points, label = "ranking of " + label_i)
    plt.legend(fontsize=30) #Ajouter une légende à la l'image
    plt.xlim(-500,500)
    plt.xlabel("Ranking", fontsize=30) #Label pour l'axe des x
    plt.ylabel("Number of occurence", fontsize=30) #Label pour l'axe des x
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30) #Taille des valeurs des axes
    plt.savefig('../Figures/Rankings.png') #Sauvergarder l'image
    plt.show() #Montrer l'image
    

def plotRankingsDistribution(players_rankings, ranking_limit = 500):
    plt.figure("Rankings distribution", figsize = (20,15)) 
    plt.title("Density per ranking", fontsize=30)
    plt.grid()
    min_rank = int(min(np.nanmin(players_rankings[0]), np.nanmin(players_rankings[1])))
    max_rank = min(int(max(np.nanmax(players_rankings[0]), np.nanmax(players_rankings[1]))),ranking_limit)
    x_points = np.arange(min_rank, max_rank)
    y_points = np.zeros(len(x_points))
    y_nan = 0
    for i in range(2):
        for ranking in players_rankings[i]:
            if np.isnan(ranking) == False and ranking < ranking_limit:
                ranking = int(ranking)
                y_points[np.where(x_points == ranking)] += 1
            else :
                y_nan += 1 
    y_points = y_points / sum(y_points)
    plt.plot(x_points, y_points, label = "Ranking of players")
    y_points_smooth = savgol_filter(y_points, 51, 3)
    plt.plot(x_points, y_points_smooth, label = "Ranking of players smoothed")
    
    alpha = 1.0
    l1 = 60.0
    l2 = 100.0
    y_points_sum_exponential = alpha* 1/l1 * np.exp(-(x_points-1)/l1) + (1-alpha)* 1/l2 * np.exp(-(x_points-1)/l2)
    plt.plot(x_points, y_points_sum_exponential, label = "Ranking of players exponential")
    
    
    plt.legend(fontsize=30) #Ajouter une légende à la l'image
    plt.xlim(-10,500)
    plt.xlabel("ATP Ranking", fontsize=30) #Label pour l'axe des x
    plt.ylabel("PDF", fontsize=30) #Label pour l'axe des x
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30) #Taille des valeurs des axes
    plt.savefig('../Figures/RankingPDF.png') #Sauvergarder l'image
    plt.show() #Montrer l'image  
    return x_points, y_points_smooth


def plotSingularValues(singular_values, singular_values_centered, completion_method, clipping):
    label = completion_method[0]
#    parameters = completion_method[1]
    n_singular_values = len(singular_values)
    plt.figure("Singular values " + label, figsize = (10,8)) 
    plt.title(label, fontsize=30)
    x_points = np.arange(1,n_singular_values+1)
    if clipping == True:
        label += " clip"
    else:
        plt.grid()

#    if parameters != None:
#        label += ", " + str(parameters)
    plt.scatter(x_points, singular_values, label = label)
    plt.scatter(x_points, singular_values_centered, label = label + " (centered)")
#    plt.legend(fontsize=30) #Ajouter une légende à la l'image
    plt.xlabel("$i^{th}$ biggest $\sigma$", fontsize=30) #Label pour l'axe des x
    plt.ylabel("$\sigma$", fontsize=30) #Label pour l'axe des x
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30) #Taille des valeurs des axes
    plt.xlim((0,n_singular_values))
    plt.yscale("log")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(list(plt.xticks()[0])[1:] + [1])
    plt.tight_layout()
    plt.savefig('../Figures/SingularValues' + completion_method[0] +'.png') #Sauvergarder l'image
    plt.show() #Montrer l'image  
    
    
def plotELODistribution(elo, elo_distribution):
    plt.figure("ELO distribution", figsize = (20,15)) 
    plt.title("PDF ELO", fontsize=30)
    plt.grid()
    plt.plot(elo, elo_distribution, label = "ELO distribution")
    
    mu_log = 1750
    sigma = 95
    elo_logistic = np.exp(-(elo-mu_log)/sigma)/(sigma * (1 + np.exp(-(elo-mu_log)/sigma))**2)
    plt.plot(elo, elo_logistic, label = "ELO distribution logistic")
    
    mu_gum = 1750
    lambd = 1/140
    elo_gumbel = lambd*np.exp(-(lambd*(elo-mu_gum) + np.exp(-(lambd*(elo-mu_gum))) ))
    plt.plot(elo, elo_gumbel, label = "ELO distribution Gumbel")
    
    plt.legend(fontsize=30) #Ajouter une légende à la l'image
    plt.xlabel("ELO", fontsize=30) #Label pour l'axe des x
    plt.ylabel("PDF", fontsize=30) #Label pour l'axe des x
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30) #Taille des valeurs des axes
    plt.savefig('../Figures/ELODistribution.png') #Sauvergarder l'image
    plt.show() #Montrer l'image  
    
    
def plotDifferenceRanking(players_rankings):
    plt.figure("Wins and difference rankings", figsize = (20,15)) 
    plt.title("Difference of rankings", fontsize=30)
    plt.grid()
    min_rank = int(np.nanmin(players_rankings[2]))
    max_rank = int(np.nanmax(players_rankings[2]))
    abs_max_rank = max(abs(min_rank), abs(max_rank))
    x_points = np.arange(-abs_max_rank, abs_max_rank)
    y_points_wins = np.zeros(len(x_points))
    y_points_loss = np.zeros(len(x_points))
    y_points_total = np.zeros(len(x_points))
    y_nan = 0
    for diff_ranking in players_rankings[2]:
        if np.isnan(diff_ranking) == False:
            diff_ranking = int(diff_ranking)
            y_points_wins[np.where(x_points == diff_ranking)] += 1
            y_points_loss[np.where(x_points == -diff_ranking)] += 1
            y_points_total[np.where(x_points == diff_ranking)] += 1
            y_points_total[np.where(x_points == -diff_ranking)] += 1
        else :
            y_nan += 1 
    nonzero_indices = np.where(y_points_total > 0)
    x_points_ratio = x_points[nonzero_indices]
    y_points_ratio = y_points_wins[nonzero_indices]/y_points_total[nonzero_indices]
    variance_beta = (y_points_wins[nonzero_indices] + 1)*(y_points_total[nonzero_indices] - y_points_wins[nonzero_indices] + 1)/((y_points_total[nonzero_indices] + 2)**2 * (y_points_total[nonzero_indices] + 3))
    std_beta = np.sqrt(variance_beta)
    s = 7.0
    l = 0.4
    y_points_sigmoid = 1.0/(1.0 + np.exp(-np.sign(x_points_ratio)*(np.abs(x_points_ratio))**(l) / s))
#    plt.plot(x_points, y_points_wins, label = "difference ranking of winner")
#    plt.plot(x_points, y_points_loss, label = "difference ranking of loser")
#    plt.plot(x_points, y_points_total, label = "difference ranking of match")
    plt.plot(x_points_ratio, y_points_ratio, label = "win percentage")
    plt.fill_between(x_points_ratio, y_points_ratio - std_beta, y_points_ratio + std_beta, alpha=0.5)
    
    plt.plot(x_points_ratio, y_points_sigmoid, label = "win percentage modified sigmoid")
    plt.legend(fontsize=30) #Ajouter une légende à la l'image
    plt.xlim(-500,500)
    plt.ylim(0.0,1.0)
    plt.xlabel("ATP Ranking Difference", fontsize=30) #Label pour l'axe des x
    plt.ylabel("Win percentage", fontsize=30) #Label pour l'axe des x
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30) #Taille des valeurs des axes
    plt.savefig('../Figures/RankingsDifference.png') #Sauvergarder l'image
    plt.show() #Montrer l'image  
    
    plt.figure("Wins and difference ELO", figsize = (20,15)) 
    plt.title("Difference of ELO ratings", fontsize=30)
    plt.grid()
    min_ELO = int(np.nanmin(players_rankings[5]))
    max_ELO = int(np.nanmax(players_rankings[5]))
    abs_max_ELO = max(abs(min_ELO), abs(max_ELO))
    x_points = np.arange(-abs_max_ELO, abs_max_ELO)
    y_points_wins = np.zeros(len(x_points))
    y_points_loss = np.zeros(len(x_points))
    y_points_total = np.zeros(len(x_points))
    y_nan = 0
    for diff_ELO in players_rankings[5]:
        if np.isnan(diff_ranking) == False:
            diff_ranking = int(diff_ranking)
            y_points_wins[np.where(x_points == diff_ELO)] += 1
            y_points_loss[np.where(x_points == -diff_ELO)] += 1
            y_points_total[np.where(x_points == diff_ELO)] += 1
            y_points_total[np.where(x_points == -diff_ELO)] += 1
        else :
            y_nan += 1 
    nonzero_indices = np.where(y_points_total > 0)
    x_points_ratio = x_points[nonzero_indices]
    y_points_ratio = y_points_wins[nonzero_indices]/y_points_total[nonzero_indices]
    variance_beta = (y_points_wins[nonzero_indices] + 1)*(y_points_total[nonzero_indices] - y_points_wins[nonzero_indices] + 1)/((y_points_total[nonzero_indices] + 2)**2 * (y_points_total[nonzero_indices] + 3))
    std_beta = np.sqrt(variance_beta)
    s2 = 250.0
    y_points_sigmoid = 1.0/(1.0 + np.exp(-x_points_ratio/ s2))
#    plt.plot(x_points, y_points_wins, label = "difference ranking of winner")
#    plt.plot(x_points, y_points_loss, label = "difference ranking of loser")
#    plt.plot(x_points, y_points_total, label = "difference ranking of match")
    plt.plot(x_points_ratio, y_points_ratio, label = "win percentage")
    plt.fill_between(x_points_ratio, y_points_ratio - std_beta, y_points_ratio + std_beta, alpha=0.5)
    
    plt.plot(x_points_ratio, y_points_sigmoid, label = "win percentage sigmoid")
    plt.legend(fontsize=30) #Ajouter une légende à la l'image
    plt.xlim(-1000,1000)
    plt.ylim(0.0,1.0)
    plt.xlabel("ELO rating Difference", fontsize=30) #Label pour l'axe des x
    plt.ylabel("Win percentage", fontsize=30) #Label pour l'axe des x
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30) #Taille des valeurs des axes
    plt.savefig('../Figures/RankingsDifferenceELO.png') #Sauvergarder l'image
    plt.show() #Montrer l'image  


def plotVaryRank(imposed_ranks, accuracy_percentages, rmse, wrmse, method):
    if np.isnan(np.sum(rmse)) == True:
        f = plt.figure("Imposed rank " + method, figsize = (10,8)) 
        f.subplots_adjust(wspace=0.2, hspace=0.4)
        plt.rc('axes', titlesize=20)
        f.suptitle(method, fontsize=30)
        
        a1 = f.add_subplot(1, 1, 1)
        a1.plot(imposed_ranks, accuracy_percentages[:,0], label = "no clip")
        a1.plot(imposed_ranks, accuracy_percentages[:,1], label = "clip")
        a1.grid()
        a1.set_ylabel("Prediction Accuracy", fontsize=30)
        a1.set_xlabel("Imposed rank", fontsize=30)
        a1.tick_params(axis = 'both', which = 'both',labelsize = 30)
    
    else:
        f = plt.figure("Imposed rank " + method, figsize = (15,15)) 
        f.subplots_adjust(wspace=0.2, hspace=0.4)
        plt.rc('axes', titlesize=20)
        f.suptitle(method, fontsize=30)
        
        a1 = f.add_subplot(3, 1, 1)
        a1.plot(imposed_ranks, accuracy_percentages[:,0], label = "no clip")
        a1.plot(imposed_ranks, accuracy_percentages[:,1], label = "clip")
        a1.grid()
        a1.set_ylabel("Prediction Accuracy", fontsize=30)
        a1.tick_params(axis = 'both', which = 'both',labelsize = 30)
        
        a2 = f.add_subplot(3, 1, 2)
        a2.plot(imposed_ranks, rmse[:,0])
        a2.plot(imposed_ranks, rmse[:,1])
        a2.grid()
        a2.set_ylabel("RMSE",fontsize=30)
        a2.tick_params(axis = 'both', which = 'both',labelsize = 30)
        
        a3 = f.add_subplot(3, 1, 3)
        a3.plot(imposed_ranks, wrmse[:,0])
        a3.plot(imposed_ranks, wrmse[:,1])
        a3.grid()
        a3.set_ylabel("WRMSE",fontsize=30)
        a3.set_xlabel("Imposed rank (before constraining)", fontsize=30)
        a3.tick_params(axis = 'both', which = 'both',labelsize = 30)
    
    lines_labels = [ax.get_legend_handles_labels() for ax in f.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    f.legend(lines, labels, fontsize=30)
    f.tight_layout()
    f.savefig('../Figures/ImposedRank' + method + '.png')
    f.show()
    

def plotVaryTau(imposed_tau, accuracy_percentages, rmse, wrmse, method):
    if np.isnan(np.sum(rmse)) == True:
        f = plt.figure("Imposed tau " + method, figsize = (15,10)) 
        f.subplots_adjust(wspace=0.2, hspace=0.4)
        plt.rc('axes', titlesize=20)
        f.suptitle(method, fontsize=30)
    
        a1 = f.add_subplot(1, 1, 1)
        a1.plot(imposed_tau, accuracy_percentages[:,0], label = "no clip")
        a1.plot(imposed_tau, accuracy_percentages[:,1], label = "clip")
        a1.grid()
        a1.set_ylabel("Prediction Accuracy", fontsize=30)
        a1.set_xlabel("Imposed tau", fontsize=30)
        a1.set_xscale('log')
        a1.tick_params(axis = 'both', which = 'both',labelsize = 30)
    
    else:
        f = plt.figure("Imposed tau " + method, figsize = (15,15)) 
        f.subplots_adjust(wspace=0.2, hspace=0.4)
        plt.rc('axes', titlesize=20)
        f.suptitle(method, fontsize=30)
    
        a1 = f.add_subplot(3, 1, 1)
        a1.plot(imposed_tau, accuracy_percentages[:,0], label = "no clip")
        a1.plot(imposed_tau, accuracy_percentages[:,1], label = "clip")
        a1.grid()
        a1.set_ylabel("Prediction Accuracy", fontsize=30)
        a1.set_xscale('log')
        a1.tick_params(axis = 'both', which = 'both',labelsize = 30)
    
        a2 = f.add_subplot(3, 1, 2)
        a2.plot(imposed_tau, rmse[:,0])
        a2.plot(imposed_tau, rmse[:,1])
        a2.grid()
        a2.set_ylabel("RMSE",fontsize=30)
        a2.set_xscale('log')
        a2.tick_params(axis = 'both', which = 'both',labelsize = 30)
    
        a3 = f.add_subplot(3, 1, 3)
        a3.plot(imposed_tau, wrmse[:,0])
        a3.plot(imposed_tau, wrmse[:,1])
        a3.grid()
        a3.set_xlabel("Imposed tau", fontsize=30)
        a3.set_ylabel("WRMSE",fontsize=30)
        a3.set_xscale('log')
        a3.tick_params(axis = 'both', which = 'both',labelsize = 30)
    
    lines_labels = [ax.get_legend_handles_labels() for ax in f.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    f.legend(lines, labels, fontsize=30)
    
    f.savefig('../Figures/ImposedTau' + method + '.png')
    f.show()


def plotVaryK(imposed_k, accuracy_percentages, rmse, wrmse, method):
    f = plt.figure("Imposed k " + method, figsize = (15,15)) 
    f.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.rc('axes', titlesize=20)
    f.suptitle(method, fontsize=30)
    
    a1 = f.add_subplot(3, 1, 1)
    a1.plot(imposed_k, accuracy_percentages[:,0], label = "no clip")
    a1.plot(imposed_k, accuracy_percentages[:,1], label = "clip")
    a1.grid()
    a1.set_ylabel("Prediction Accuracy", fontsize=30)
    a1.tick_params(axis = 'both', which = 'both',labelsize = 30)
    
    a2 = f.add_subplot(3, 1, 2)
    a2.plot(imposed_k, rmse[:,0])
    a2.plot(imposed_k, rmse[:,1])
    a2.grid()
    a2.set_ylabel("RMSE",fontsize=30)
    a2.tick_params(axis = 'both', which = 'both',labelsize = 30)
    
    a3 = f.add_subplot(3, 1, 3)
    a3.plot(imposed_k, wrmse[:,0])
    a3.plot(imposed_k, wrmse[:,1])
    a3.grid()
    a3.set_ylabel("WRMSE",fontsize=30)
    a3.set_xlabel("Imposed k", fontsize=30)
    a3.tick_params(axis = 'both', which = 'both',labelsize = 30)
    
    lines_labels = [ax.get_legend_handles_labels() for ax in f.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    f.legend(lines, labels, fontsize=30)
    
    f.savefig('../Figures/ImposedK' + method + '.png')
    f.show() 
    
    
def plotVaryB(imposed_b, accuracy_percentages, rmse, wrmse, probability_model, estimator):
    if np.isnan(np.sum(rmse)) == True:
        if estimator == None:
            f = plt.figure("Imposed b " + probability_model, figsize = (15,10)) 
            f.subplots_adjust(wspace=0.2, hspace=0.4)
            plt.rc('axes', titlesize=20)
            f.suptitle(probability_model, fontsize=30)
        else:
            f = plt.figure("Imposed b " + probability_model+ estimator, figsize = (15,10)) 
            f.subplots_adjust(wspace=0.2, hspace=0.4)
            plt.rc('axes', titlesize=20)
            f.suptitle(probability_model + " "+estimator, fontsize=30)
        
        a1 = f.add_subplot(1, 1, 1)
        a1.plot(imposed_b, accuracy_percentages[:,0], label = "no clip")
        a1.plot(imposed_b, accuracy_percentages[:,1], label = "clip")
        a1.grid()
        a1.set_ylabel("Prediction Accuracy", fontsize=30)
        a1.set_xlabel("Imposed $b$", fontsize=30)
        a1.tick_params(axis = 'both', which = 'both',labelsize = 30)
    
    else:
        f = plt.figure("Imposed b " + probability_model+ estimator, figsize = (15,15)) 
        f.subplots_adjust(wspace=0.2, hspace=0.4)
        plt.rc('axes', titlesize=20)
        f.suptitle(probability_model +" "+ estimator, fontsize=30)
    
        a1 = f.add_subplot(3, 1, 1)
        a1.plot(imposed_b, accuracy_percentages[:,0], label = "no clip")
        a1.plot(imposed_b, accuracy_percentages[:,1], label = "clip")
        a1.grid()
        a1.set_ylabel("Prediction Accuracy", fontsize=30)
        a1.tick_params(axis = 'both', which = 'both',labelsize = 30)
    
        a2 = f.add_subplot(3, 1, 2)
        a2.plot(imposed_b, rmse[:,0])
        a2.plot(imposed_b, rmse[:,1])
        a2.grid()
        a2.set_ylabel("RMSE",fontsize=30)
        a2.tick_params(axis = 'both', which = 'both',labelsize = 30)
    
        a3 = f.add_subplot(3, 1, 3)
        a3.plot(imposed_b, wrmse[:,0])
        a3.plot(imposed_b, wrmse[:,1])
        a3.grid()
        a3.set_xlabel("Imposed $b$", fontsize=30)
        a3.set_ylabel("WRMSE",fontsize=30)
        a3.tick_params(axis = 'both', which = 'both',labelsize = 30)
    
    if estimator == None:
        lines_labels = [ax.get_legend_handles_labels() for ax in f.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        f.legend(lines, labels, fontsize=30)
    
        f.savefig('../Figures/ImposedB' + probability_model+'.png')
        f.show()
        
    else:
        lines_labels = [ax.get_legend_handles_labels() for ax in f.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        f.legend(lines, labels, fontsize=30)
    
        f.savefig('../Figures/ImposedB' + probability_model + estimator + '.png')
        f.show()
    
    
def plotVaryLambda(imposed_lambda, accuracy_percentages, method):
    plt.figure("Imposed lambda " + method, figsize = (10,8)) 
    plt.title(method, fontsize=30)
    plt.plot(imposed_lambda, accuracy_percentages[:,0], label = "no clip")
    plt.plot(imposed_lambda, accuracy_percentages[:,1], label = "clip")
    plt.grid()
    plt.xscale('log')
    plt.xlabel("Imposed $\lambda$", fontsize=30)
    plt.ylabel("Prediction Accuracy", fontsize=30)
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30)
    plt.legend(fontsize=30)
    plt.tight_layout()
    plt.savefig('../Figures/ImposedLambda' + method + '.png')
    plt.show()
    
    
def plotVaryS(imposed_s, accuracy_percentages, method):
    plt.figure("Imposed s " + method, figsize = (10,8)) 
    plt.title(method, fontsize=30)
    plt.plot(imposed_s, accuracy_percentages[:,0], label = "no clip")
    plt.plot(imposed_s, accuracy_percentages[:,1], label = "clip")
    plt.grid()
    plt.xscale('log')
    plt.xlabel("Imposed $s$", fontsize=30)
    plt.ylabel("Prediction Accuracy", fontsize=30)
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30)
    plt.legend(fontsize=30)
    plt.tight_layout()
    plt.savefig('../Figures/Imposeds' + method + '.png')
    plt.show()
    
    
def plotStdE(imposed_s, e_matrix, method):
    e_matrix_mean = np.mean(e_matrix, axis=2).T
    e_matrix_sigma = np.std(e_matrix, axis=(1,2))
    
    plt.figure("Std E " + method, figsize = (10,8)) 
    plt.title(method, fontsize=30)
    plt.plot(imposed_s, e_matrix_sigma)
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Imposed $s$", fontsize=30)
    plt.ylabel("Standard deviation of $E$", fontsize=30)
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30)
    plt.tight_layout()
    plt.savefig('../Figures/StdE' + method + '.png')
    plt.show()
    
    return e_matrix_mean, e_matrix_sigma

def plotVariabilityRuns(runs_indices, accuracy, method):
    plt.figure("Variability", figsize = (20,15)) 
    plt.title("Variability of runs", fontsize=30)
    plt.grid()
    plt.plot(runs_indices, accuracy[:,0], label = method + " no clip")
    plt.plot(runs_indices, accuracy[:,1], label = method + " clip")
    plt.legend(fontsize=30) #Ajouter une légende à la l'image
    plt.xlabel("Run index", fontsize=30) #Label pour l'axe des x
    plt.ylabel("Prediction accuracy", fontsize=30) #Label pour l'axe des x
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30) #Taille des valeurs des axes
    plt.savefig('../Figures/VariabilityAccuracy.png') #Sauvergarder l'image
    plt.show() #Montrer l'image  
    
    
def plotAccuracyDifferentMethods(methods, accuracy_percentages, accuracy_upper_bound, accuracy_known):
    plt.figure("Accuracy Methods", figsize = (20,15)) 
    plt.title("Comparison between methods", fontsize=30)
    plt.grid()
    for i in range(len(methods)):
        method_name = methods[i][0]
        method_parameter = methods[i][1]
        label = method_name
        if method_parameter != None:
            label += ", " + str(method_parameter)
        plt.plot(np.arange(len(accuracy_percentages[i])), accuracy_percentages[i], label = label)
    plt.plot(np.arange(len(accuracy_percentages[i])), accuracy_upper_bound*np.ones(len(accuracy_percentages[i])), label = "Upper bound : clipping equality")
    plt.plot(np.arange(len(accuracy_percentages[i])), accuracy_known*np.ones(len(accuracy_percentages[i])), label = "Exact reconstruction")
    plt.legend(fontsize=30) #Ajouter une légende à la l'image
    plt.xlabel("No specific axis", fontsize=30) #Label pour l'axe des x
    plt.ylabel("Mean accuracy percentage", fontsize=30) #Label pour l'axe des x
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30) #Taille des valeurs des axes
    plt.savefig('../Figures/AccuracyComparisonClipping.png') #Sauvergarder l'image
    plt.show() #Montrer l'image  


def plotRMSEDifferentMethods(methods, rmse):
    plt.figure("RMSE Methods", figsize = (20,15)) 
    plt.title("Comparison between methods", fontsize=30)
    plt.grid()
    for i in range(len(methods)):
        method_name = methods[i][0]
        method_parameter = methods[i][1]
        label = method_name
        if method_parameter != None:
            label += ", " + str(method_parameter)
        plt.plot(np.arange(len(rmse[i])), rmse[i], label = label)
    plt.legend(fontsize=30) #Ajouter une légende à la l'image
    plt.xlabel("No specific axis", fontsize=30) #Label pour l'axe des x
    plt.ylabel("RMSE", fontsize=30) #Label pour l'axe des x
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30) #Taille des valeurs des axes
    plt.savefig('../Figures/RMSEComparisonClipping.png') #Sauvergarder l'image
    plt.show() #Montrer l'image  
    
    
def plotWRMSEDifferentMethods(methods, wrmse):
    plt.figure("WRMSE Methods", figsize = (20,15)) 
    plt.title("Comparison between methods", fontsize=30)
    plt.grid()
    for i in range(len(methods)):
        method_name = methods[i][0]
        method_parameter = methods[i][1]
        label = method_name
        if method_parameter != None:
            label += ", " + str(method_parameter)
        plt.plot(np.arange(len(wrmse[i])), wrmse[i], label = label)
    plt.legend(fontsize=30) #Ajouter une légende à la l'image
    plt.xlabel("No specific axis", fontsize=30) #Label pour l'axe des x
    plt.ylabel("WRMSE", fontsize=30) #Label pour l'axe des x
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30) #Taille des valeurs des axes
    plt.savefig('../Figures/WRMSEComparisonClipping.png') #Sauvergarder l'image
    plt.show() #Montrer l'image  