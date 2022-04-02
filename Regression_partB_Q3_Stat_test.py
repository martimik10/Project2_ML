from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import sklearn.tree
import scipy.stats
import numpy as np, scipy.stats as st
import pandas as pd

# requires data from exercise 1.5.1
from Regression_Q2_V2 import yhat1, y_true
from Regression_PartB_Q1_Baseline import yhat2
from Regression_PartB_ANN_V2 import yhat3


print("yhat1", yhat1)    #Regression
print("yhat2", yhat2)    #Baseline
print("yhat3", yhat3)    #ANN
print("y_true", y_true)   #True values





# perform statistical comparison of the models
# compute z with squared error.
zA = np.abs(y_true - yhat2 ) ** 2

# compute confidence interval of model A
alpha = 0.05
CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
zB = np.abs(y_true - yhat1 ) ** 2
z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

print("CI", np.round(CI, 2), "   pval", p)