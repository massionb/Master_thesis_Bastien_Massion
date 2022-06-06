#
# LINMA2990 - Master Thesis
# The mathematics of Tennis Ranking: Game outcome prediction by low-rank methods
#
# Data extraction for ATP tournaments (2000-2016)
# 
# Bastien Massion, Julien Herman
# Created on 01/11/2021
#
# Last version on 01/03/2022
#

#CSV data
import csv as csv

#Manipulate data
import numpy as np

#Deep copy
import copy as cop

#Datetimes
from datetime import datetime



# Imports the data from a CSV datafile
# Returns labels, raw data and number of labels
def importData(filename):
    file = open(filename, encoding='utf-8')
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    return header, rows

# Removes wrong sample if a sample has not the right length
def removeNonConformLengthData(header,data):
    nFeatures = len(header)
    nData = len(data)
    nRemoved = 0 
    for i in range(nData):
        if len(data[i-nRemoved]) != nFeatures :
            print("Problem non conform data :", data[i-nRemoved])
            del(data[i-nRemoved])
            nRemoved += 1
    if nRemoved > 0:
        print("Number of non conform data removed =",nRemoved)
    return data


# Returns True if s is an integer, False otherwise
def isInt(s):
    try:
        s = int(s)
        return True
    except ValueError:
        return False
    
# Returns True if s is a float, False otherwise
def isFloat(s):
    try:
        s = float(s)
        return True
    except ValueError:
        return False
    
# Returns True if s is a boolean, False otherwise
def isBool(s):
    if s == 'True' or s == 'False' :
        return True
    else:
        return False
    
# Returns True if s is a date, False otherwise
def isDate(s):
    try: 
        s = datetime.strptime(s, '%d/%m/%Y')
        return True
    except ValueError:
        return False
    

# Extracts the datatypes of each label from the first row of data
# Assumption : first row of data is in right format
# Assumption2 : all elements are originally in String format
def getDataType(data,header):
    nHeader = len(header)
    nData = len(data)
    
    types = []
    for i in range(nHeader):
        types.append('Undetermined')
    
    for i in range(nHeader):
        j = 0
        while (types[i] == 'Undetermined' or types[i] == 'Int') and j < nData:
            elem = data[j][i]
            if elem != '':
                if isInt(elem):
                    types[i] = 'Int'
                elif isFloat(elem):
                    types[i] = 'Float'
                elif isBool(elem):
                    types[i] = 'Bool'
                elif isDate(elem):
                    types[i] = 'Date'
                else :
                    types[i] = 'String'
            j += 1
    
    return types


# Transforms the list of strings (data) into a list of good formats
# Removes wrong sample if a wrong format element is detected in this sample
def transformDataToTypes(data,types):
    for j in range(len(types)):
        typej = types[j]
        if typej == 'Int':
            nRemoved = 0 
            for i in range(len(data)):
                if data[i-nRemoved][j] == '':
                    data[i-nRemoved][j] = None
                elif isInt(data[i-nRemoved][j]):
                    data[i-nRemoved][j] = int(data[i-nRemoved][j])
                else:
                    print("Problem Int : data[%d,%d]" %(i,j), data[i-nRemoved][j])
                    del(data[i-nRemoved])
                    nRemoved += 1
        if typej == 'Float':
            nRemoved = 0 
            for i in range(len(data)):
                if data[i-nRemoved][j] == '':
                    data[i-nRemoved][j] = None
                elif isFloat(data[i-nRemoved][j]):
                    data[i-nRemoved][j] = float(data[i-nRemoved][j])
                else:
                    print("Problem Float : data[%d,%d] =" %(i,j), data[i-nRemoved][j])
                    del(data[i-nRemoved])
                    nRemoved += 1
        if typej == 'Bool':
            nRemoved = 0 
            for i in range(len(data)):
                if data[i-nRemoved][j] == '':
                    data[i-nRemoved][j] = None
                elif isBool(data[i-nRemoved][j]):
                    data[i-nRemoved][j] = eval(data[i-nRemoved][j])
                else:
                    print("Problem Bool : data[%d,%d]" %(i,j), data[i-nRemoved][j])
                    del(data[i-nRemoved])
                    nRemoved += 1
        if typej == 'Date':
            nRemoved = 0 
            for i in range(len(data)):
                if data[i-nRemoved][j] == '':
                    data[i-nRemoved][j] = None
                elif isDate(data[i-nRemoved][j]):
                    data[i-nRemoved][j] = datetime.strptime(data[i-nRemoved][j], '%d/%m/%Y')
                else:
                    print("Problem Date : data[%d,%d]" %(i,j), data[i-nRemoved][j])
                    del(data[i-nRemoved])
                    nRemoved += 1
        if typej == 'String':
            nRemoved = 0 
            for i in range(len(data)):
                if data[i-nRemoved][j] == '':
                    data[i-nRemoved][j] = None
                elif data[i-nRemoved][j] == '' :
                    print("Problem String : data[%d,%d]" %(i,j), data[i-nRemoved][j])
                    del(data[i-nRemoved])
                    nRemoved += 1
    return data

# Transforms data in right format and removes problematic data
def prepareData(header, data):
    data = removeNonConformLengthData(header,data)
    types = getDataType(data,header)
    data = transformDataToTypes(data,types)
    return data, types


# Separates the class (label y) from the other features (label x)
def separateClassAndIndexFromFeatures(header,types,data,dataValues,dataCategories):
    n = len(data)
    featuresindices = list(range(len(header)))
    
    try :
        dataindex = header.index('')
    except ValueError:
        dataindex = None
        dataIndex = None
    try :
        yindex = header.index('y')
    except ValueError:
        yindex = None
        dataClass = None
        dataClassType = None
        dataClassCategory = None
        dataClassValues = None
        headerClass = None
        
    if dataindex != None:
        dataIndex = [data[i][dataindex] for i in range(n)]
        featuresindices.remove(dataindex)
        
    if yindex != None:
        dataClassValues = dataValues[yindex]
        dataClassCategory = dataCategories[yindex]
        dataClassType = types[yindex]
        dataClass = [data[i][yindex] for i in range(n)]
        headerClass = header[yindex]
        featuresindices.remove(yindex)
    
    dataFeatures = [[data[j][i] for i in featuresindices] for j in range(n)]
    dataFeaturesValues = [dataValues[i] for i in featuresindices]
    dataFeaturesTypes = [types[i] for i in featuresindices]
    dataFeaturesCategories = [dataCategories[i] for i in featuresindices]
    headerFeatures = [header[i] for i in featuresindices]
    return headerFeatures, dataFeatures, dataFeaturesTypes, dataFeaturesValues, dataFeaturesCategories, dataIndex, headerClass, dataClass, dataClassType, dataClassValues, dataClassCategory
   
# Returns a list of lists containing the indices of the not-None values for each of the features
def notNoneIndices(data,types):
    notNoneData = []
    for i in range(len(types)):
        notNoneFeature = []
        for j in range(len(data)):
            if data[j][i] != None:
                notNoneFeature.append(j)
        notNoneData.append(notNoneFeature)
    return notNoneData

# Returns a first list of different values possible for each feature 
# Returns a second list with True for discrete features (limited amount of possible values) and False for continuous features (large amount of possible values)
# Discrete features need classification and continuous features need regression
def listValuesForFeature(data,indices,types):
    featuresValues = []
    categories = []
    wiggleRoom = 0.01    #Percentage of values above which we consider a continuous feature
    
    for j in range(len(types)):
        featurejValues = []
        for i in indices[j]:
            try :
                featurejValues.index(data[i][j])
            except :
                featurejValues.append(data[i][j])
                
        featurejValues.sort()
        featuresValues.append(featurejValues)
           
        if len(featurejValues) >= len(indices[j])*wiggleRoom:
            categories.append(False)
        else:
            categories.append(True)
            
    return featuresValues, categories

# Replaces true String classes features by integers for classes, and boolean features by 0 and 1
def replaceClassesByNumbers(data, indices, dataTypes, dataValues, dataCategories):
    dataNumbers = cop.deepcopy(data)
    dataValuesNumbers = cop.deepcopy(dataValues)
    dataTypesNumbers = dataTypes.copy()
    for j in range(len(dataTypes)):
        if dataTypes[j] == 'String' or dataTypes[j] == 'Bool' :
            if dataCategories[j] == True:
                for i in indices[j]:
                    dataNumbers[i][j] = dataValues[j].index(data[i][j])
                dataValuesNumbers[j] = list(range(len(dataValues[j])))
                dataTypesNumbers[j] = 'Int'
            else:
                print("Problem : feature with too much different strings")
    return dataNumbers, dataValuesNumbers, dataTypesNumbers


# Transforms a list of list of numbers in numpy array, returns None if listPython is None
def toNumpyArray(listPython):
    if listPython == None:
        return None
    else:
        return np.array(listPython)
    
# Gives all data in good format from filename
def manageData(filename):
    header, data = importData(filename)
    data, dataTypes = prepareData(header, data)
    notNoneData = notNoneIndices(data, dataTypes)
    dataValues, dataCategories = listValuesForFeature(data, notNoneData, dataTypes)
    players = sorted(list(set(dataValues[header.index('Winner')] + dataValues[header.index('Loser')])))
    n_players = len(players)
    players_indices = list(range(n_players))
    return (data, header, dataTypes, dataValues, dataCategories, players, n_players, players_indices)
