# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 19:30:47 2022

@author: Marti
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit

from Regress_prep_loo import *
#https://inria.github.io/scikit-learn-mooc/python_scripts/linear_models_regularization.html
# alphas = np.logspace(-2, 0, num=20)
alphas = np.logspace(-3, 2.5, num=150)
# print(alphas)
ridge = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(),
                      RidgeCV(alphas=alphas, store_cv_values=True))

cv = ShuffleSplit(n_splits=10, random_state=1)
cv_results = cross_validate(ridge, X, y,
                            cv=cv, scoring="neg_mean_squared_error",
                            return_train_score=True,
                            return_estimator=True, n_jobs=2)

test_error = -cv_results["test_score"]
print(f"Mean squared error of linear regression model on the test set:\n"
      f"{test_error.mean():.3f} +/- {test_error.std():.3f}")

######try to get coeffitients

#####      

#By optimizing alpha, we see that the training and testing scores are close.
# It indicates that our model is not overfitting.
mse_alphas = [est[-1].cv_values_.mean(axis=0)
              for est in cv_results["estimator"]]
cv_alphas = pd.DataFrame(mse_alphas, columns=alphas)


plt.xscale("log")
# cv_alphas.mean(axis=0).plot(marker="+")
cv_alphas.mean(axis=0).plot(linewidth=3) #PLOT funkcni

best_alphas = [est[-1].alpha_ for est in cv_results["estimator"]]
#TODO: zelenou tecku udelat spravne dolu tzn pouzit spravny alfy tzn k cemu je cv_alphas a k cemu jsou best_alphas
print("best alphas", best_alphas)
x_marker = np.round(np.mean(best_alphas))
y_marker = 5.045 #nejnizsi error, my ale bereme pak average vsech nejlepsich v ramci vsech fold validaci test_error.mean()
plt.text(x_marker-97, y_marker-0.005, '({}, {})'.format(x_marker, y_marker))
plt.plot(x_marker, y_marker, marker="o", markersize=10, markeredgecolor="green", markerfacecolor="green")

plt.ylabel("Generalization error")
plt.xlabel("λ")
_ = plt.title("Generalization error obtained by two-layer cross-validation")
legend(['Gen. Error','Lowest error'])

# plt.savefig('images/regress_Q2_V2_gen_error.pdf',bbox_inches = 'tight')
print(f"The mean optimal alpha leading to the lowest generalization error is:\n"
      f"{np.mean(best_alphas):.2f} +/- {np.std(best_alphas):.2f}")
print("POZO POUZIVAME ZDE POLYNOM, IKDYZ VYCHAZI LEPE, CHTEJI V REPORTU JEN CARU\PLANE/HYPERPLANE")

model_done = ridge.fit(X, y)

#################### TESTING ON NEVER SEEN DATA
def predict_unseen(unseen_file, element=0):
    """USAGE: Completely remove few instances from dataset, retrain model and then
       use the removed values, standardize them and
       use the predict() function to see the output, compare it to true value
       be sure to include atleast instance from one of each attribute for nice data proccessing"""
    filename = unseen_file
    df_unseen = pd.read_csv(filename)
    df_unseen = df_unseen.iloc[: , 1:] #drop "rowid"
    df_unseen = df_unseen.iloc[: , 0:7] #drop "year"

    one_hot = pd.get_dummies(df_unseen['species']) #one out of K encode species
    df_unseen = df_unseen.drop('species', axis = 1)
    df_unseen = df_unseen.join(one_hot)
    one_hot = pd.get_dummies(df_unseen['island']) 
    df_unseen = df_unseen.drop('island', axis = 1)
    df_unseen = df_unseen.join(one_hot)
    one_hot = pd.get_dummies(df_unseen['sex']) 
    df_unseen = df_unseen.drop('sex', axis = 1)
    df_unseen = df_unseen.join(one_hot)
    df_unseen.info()
    
    raw_data_unseen = df_unseen.values
    X_unseen = raw_data_unseen[:, 1:]

    y_unseen_label = raw_data_unseen[:, 0]
    X_info_unseen = df_unseen.iloc[:, 1:]
    # X_info_unseen.info()
    X_std_unseen = preprocessing.scale(X_unseen[:, 0:3]) 
    X_unseen[:, 0:3] = X_std_unseen #standardize Bill depth, FLipper, Mass
    
    predict_unseen = ridge.predict(X_unseen)
    
    abs_true_error = np.round(abs(predict_unseen -  y_unseen_label), 2)
    perc_error = np.round(abs_true_error*100/y_unseen_label, 2)
    print("Predictions of unseen data:", np.round(predict_unseen, 2))
    print("True data:", y_unseen_label)
    print("Abs True error",abs_true_error )
    print("Error in %:", perc_error)
    print("Mean true error %", round(np.mean(perc_error), 2))
    return predict_unseen, y_unseen_label
    
    
#should be 39.1, is 40.08. Coz je chyba 2.25% coz sedi (protoze to presne tohodle penguina uz znalo, jinak je avg 5.5%)

################# STATISTICAL ANALYSIS ON UNSEEN DATA

unseen_file = "dataset/penguins_testing_regression_unseen.csv"
yhat1, y_true = predict_unseen(unseen_file) 




