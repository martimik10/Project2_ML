# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplot, plot, legend, show,  xlabel, ylabel, xticks, yticks, xscale
import warnings
from sklearn import preprocessing #standardisation
# import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import category_encoders as ce

def check_std(attribute):
    attr_name = attributeNames[attribute]
    x = X[:, attribute]
    mean_x = x.mean()
    std_x = x.std(ddof=1)   #standard deviation
    
    print("checking non vs standardized mean and std of attribute:", attr_name)
    print('Mean:', round(mean_x, 2))
    print('Standard Deviation:',round(std_x, 2))
    
    x = X_std[:, attribute]
    mean_x = x.mean()
    std_x = x.std(ddof=1)   #standard deviation
    
    print('Mean:',round(mean_x, 2))
    print('Standard Deviation:', round(std_x, 2))

#############PREPARATION FOR REGRESSION############
# Dataset is  already without outliers and NaNs
# First, we load data and then use LEave One Out Encoding (on-hot)
#######################

filename = "dataset/penguins_testing_regression.csv"
# filename = "dataset/penguins_testing_regression.csv"
df = pd.read_csv(filename)
df = df.iloc[: , 1:] #drop "rowid"
df = df.iloc[: , 0:7] #drop "year"


#now we have 0 species, 1 island, 2 bill len, 3 bill dep, 4 flipper, 5 body mass, 6 sex, 7 year
one_hot = pd.get_dummies(df['species']) #one out of K encode species
df = df.drop('species', axis = 1)
df = df.join(one_hot)


one_hot = pd.get_dummies(df['island']) 
df = df.drop('island', axis = 1)
df = df.join(one_hot)

one_hot = pd.get_dummies(df['sex']) 
df = df.drop('sex', axis = 1)
df = df.join(one_hot)

X_info = df.iloc[:, 1:] #dtio Bill len as it is the output (only for showing)
X_info.info()

########################################################
#Data preprocessing - we want to predict Bill length (0th attribute) on all the others

raw_data = df.values
X = raw_data[:, 1:]

attributeNames = np.asarray(df.columns[1:])
yNames = np.asarray(df.columns[0])
classLabels = raw_data[:, [10,11]] #we want it to be sex
# classNames = ['female', 'male'] #idk if that works with one-hot
# classDict = dict(zip(classNames,range(len(classNames)))) #{'female': 0, 'male': 1, 'nan': 2}


y = raw_data[:, 0]
N, M = X.shape
# C = len(classNames)

print("!!!!We are using standardized non-cathegorical data now!!!!!")
print("- y (what we want to estimate is not standardized")


######### STANDARZDIZE
# print(X[:, 0])
X_std = preprocessing.scale(X[:, 0:3]) #standardize Bill depth, FLipper, Mass
X[:, 0:3] = X_std

# X = preprocessing.scale(X) #standardize EVERYTHING
print(X[0])

# print(X)

# ### check if mean is zero and std is 1:
# attribute = 3 #choose from attributes we want to predict by
# check_std(attribute)
