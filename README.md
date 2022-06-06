# Master_thesis_Bastien_Massion
Python codes related to the Master's thesis of Bastien Massion:

Tennis matches outcome prediction via Low-Rank approaches

EPL, UCLouvain, June 2022

# Experiments
To recreate the results of experiment called NAME from chapter NUMBER of the thesis, run the files: 

Experiment_NUMBER_NAME.py

# Organization
Experiment files call other general files which all have some particular purpose. There are listed below in the same order as they appear in the experiment files:

Data_2000_2016.csv is the dataset of ATP matches spanning from season 2000 to season 2016.

Data_thesis.py is called to import all data from the dataset.

Selection_thesis.py and Selection_ratings_thesis.py are called to select the training and testing datasets.

Statistics_thesis.py is called to compute the matrix to complete in the matrix completion problem.

Optimization_thesis.py and Optimization_levels_thesis.py are called to start the optimization process.

lmafit.py is an implementation of the LMaFit method by Hazan Deglayan and Simon Vary.

Error_thesis.py and Error_tensor_thesis.py are called to compute the error metrics.

Plotting_thesis.py is called to create the plots of the experiments.

