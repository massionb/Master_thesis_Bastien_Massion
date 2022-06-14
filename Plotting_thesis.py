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
from Statistics_thesis import *

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
    plt.legend(fontsize=20) #Ajouter une légende à la l'image
    plt.xlabel("$i^{th}$ biggest $\sigma$", fontsize=30) #Label pour l'axe des x
    plt.ylabel("$\sigma$", fontsize=30) #Label pour l'axe des x
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30) #Taille des valeurs des axes
    plt.xlim((0,n_singular_values))
    plt.yscale("log")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(list(plt.xticks()[0])[1:] + [1])
    plt.tight_layout()
    plt.savefig('SingularValues' + completion_method[0] +'.png') #Sauvergarder l'image
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
    f.savefig('ImposedRank' + method + '.png')
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
    
    f.savefig('ImposedTau' + method + '.png')
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
    
    f.savefig('ImposedK' + method + '.png')
    f.show() 
    
    
def plotVaryB(imposed_b, accuracy_percentages, rmse, wrmse, probability_model, estimators):
    if np.isnan(np.sum(rmse)) == True:
        if estimators == None:
            f = plt.figure("Imposed b " + probability_model, figsize = (15,10)) 
            f.subplots_adjust(wspace=0.2, hspace=0.4)
            plt.rc('axes', titlesize=20)
            f.suptitle(probability_model, fontsize=30)
        else:
            f = plt.figure("Imposed b " + probability_model + ": train " + estimators[0] + ", test " + estimators[1], figsize = (15,10)) 
            f.subplots_adjust(wspace=0.2, hspace=0.4)
            plt.rc('axes', titlesize=20)
            f.suptitle(probability_model + ": train " + estimators[0] + ", test " + estimators[1], fontsize=30)
        
        a1 = f.add_subplot(1, 1, 1)
        a1.plot(imposed_b, accuracy_percentages[:,0], label = "no clip")
        a1.plot(imposed_b, accuracy_percentages[:,1], label = "clip")
        a1.grid()
        a1.set_ylabel("Prediction Accuracy", fontsize=30)
        a1.set_xlabel("Imposed $b$", fontsize=30)
        a1.tick_params(axis = 'both', which = 'both',labelsize = 30)
    
    else:
        f = plt.figure("Imposed b " + probability_model + ": train " + estimators[0] + ", test " + estimators[1], figsize = (15,15)) 
        f.subplots_adjust(wspace=0.2, hspace=0.4)
        plt.rc('axes', titlesize=20)
        f.suptitle(probability_model + ": train " + estimators[0] + ", test " + estimators[1], fontsize=30)
    
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
    
    if estimators == None:
        lines_labels = [ax.get_legend_handles_labels() for ax in f.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        f.legend(lines, labels, fontsize=30)
    
        f.savefig('ImposedB' + probability_model+'.png')
        f.show()
        
    else:
        lines_labels = [ax.get_legend_handles_labels() for ax in f.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        f.legend(lines, labels, fontsize=30)
    
        f.savefig('ImposedB' + probability_model + estimators[0] + estimators[1] + '.png')
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
    plt.savefig('ImposedLambda' + method + '.png')
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
    plt.savefig('Imposeds' + method + '.png')
    plt.show()
    
    
def plotStdE(imposed_values, e_matrix, method, imposed_parameter):
    e_matrix_mean = np.mean(e_matrix, axis=2).T
    e_matrix_sigma = np.std(e_matrix, axis=(1,2))
    
    plt.figure("Std E " + method, figsize = (10,8)) 
    plt.title(method, fontsize=30)
    plt.plot(imposed_values, e_matrix_sigma, label = imposed_parameter)
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Imposed parameter, $\lambda s$ constant", fontsize=30)
    plt.ylabel("Standard deviation of $E$", fontsize=30)
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30)
    plt.legend(fontsize=30)
    plt.tight_layout()
    plt.savefig('StdE' + method + imposed_parameter + '.png')
    plt.show()
    
    return e_matrix_mean, e_matrix_sigma

def plotVariabilityRuns(runs_indices, accuracy, method):
    plt.figure("Variability", figsize = (20,15)) 
    plt.title("Variability of runs", fontsize=30)
    plt.grid()
    plt.scatter(runs_indices, accuracy[:,0], s= 300, label = method + " no clip")
    plt.scatter(runs_indices, accuracy[:,1], s= 300, label = method + " clip")
    plt.plot(runs_indices, accuracy[:,0])
    plt.plot(runs_indices, accuracy[:,1])
    plt.legend(fontsize=30) #Ajouter une légende à la l'image
    plt.xlabel("Run index", fontsize=30) #Label pour l'axe des x
    plt.ylabel("Prediction accuracy", fontsize=30) #Label pour l'axe des x
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30) #Taille des valeurs des axes
    plt.savefig('VariabilityAccuracy.png') #Sauvergarder l'image
    plt.show() #Montrer l'image  


def plotComparisonSeasons(seasons, accuracy, methods):
    plt.figure("Comparison accuracy", figsize = (20,15)) 
    plt.title("Accuracy over seasons", fontsize=30)
    plt.grid()
    n_seasons = len(accuracy[:,0])
    for i in range(n_seasons):
        plt.scatter(seasons, accuracy[i], s= 300, label = methods[i][0])
        plt.plot(seasons, accuracy[i])
    plt.legend(fontsize=30) #Ajouter une légende à la l'image
    plt.xlabel("Season", fontsize=30) #Label pour l'axe des x
    plt.ylabel("Prediction accuracy", fontsize=30) #Label pour l'axe des x
    plt.tick_params(axis = 'both', which = 'both',labelsize = 30) #Taille des valeurs des axes
    plt.savefig('ComparisonAccuracy.png') #Sauvergarder l'image
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
    plt.savefig('AccuracyComparisonClipping.png') #Sauvergarder l'image
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
    plt.savefig('RMSEComparisonClipping.png') #Sauvergarder l'image
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
    plt.savefig('WRMSEComparisonClipping.png') #Sauvergarder l'image
    plt.show() #Montrer l'image  