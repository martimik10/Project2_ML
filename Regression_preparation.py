# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplot, plot, legend, show,  xlabel, ylabel, xticks, yticks
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
# First, we load data
#######################

##############Change here############
#change columns for different regression models
y_cols = [3] #What we want to estimate: Bill length
attribute_cols = [4, 5, 6] #By what we estimate: Bill depth, flipper length, weight
classes_cols = [7]  #Only for visualization purposes, now it is sex (try species?)
#####################################
filename = "dataset/penguins.csv"
df = pd.read_csv(filename)

raw_data = df.values

X = raw_data[:, attribute_cols]
attributeNames = np.asarray(df.columns[attribute_cols])
yNames = np.asarray(df.columns[y_cols])
classLabels = raw_data[:, classes_cols] #we want it to be sex
classNames = np.unique(classLabels.astype("str")) 
classDict = dict(zip(classNames,range(len(classNames)))) #{'female': 0, 'male': 1, 'nan': 2}


y = raw_data[:, y_cols]
N, M = X.shape
C = len(classNames)

print("what we estimate by:", X[0, :],"\n     ", attributeNames)
print("What we want to estimate:", y[0, :],"\n     ", yNames)


######### STANDARZDIZE
X = preprocessing.scale(X) #standardized - USE THIS IN LATER QUESTIONS!

### check if mean is zero and std is 1:
attribute = 0 #choose from attributes we want to predict by
# check_std(attribute)