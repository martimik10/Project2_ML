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
alphas = np.logspace(-2, 2, num=50)
# print(alphas)
ridge = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(),
                      RidgeCV(alphas=alphas, store_cv_values=True))

cv = ShuffleSplit(n_splits=10, random_state=1)
cv_results = cross_validate(ridge, X, y,
                            cv=cv, scoring="neg_mean_squared_error",
                            return_train_score=True,
                            return_estimator=True, n_jobs=2)
train_error = -cv_results["train_score"]
print(f"Mean squared error of linear regression model on the train set:\n"
      f"{train_error.mean():.3f} +/- {train_error.std():.3f}")

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
# print("cv_alfas",cv_alphas)

plt.xscale("log")
# cv_alphas.mean(axis=0).plot(marker="+")
cv_alphas.mean(axis=0).plot(linewidth=3) #PLOT funkcni

best_alphas = [est[-1].alpha_ for est in cv_results["estimator"]]
#TODO: zelenou tecku udelat spravne dolu tzn pouzit spravny alfy tzn k cemu je cv_alphas a k cemu jsou best_alphas

x_marker = np.round(np.mean(best_alphas))
y_marker = 5.13 #nejnizsi error, my ale bereme pak average vsech nejlepsich v ramci vsech fold validaci test_error.mean()
plt.text(x_marker-48.5, y_marker, '({}, {})'.format(x_marker, y_marker))
plt.plot(x_marker-4, y_marker, marker="o", markersize=10, markeredgecolor="green", markerfacecolor="green")

plt.ylabel("Generalization error")
plt.xlabel("λ")
_ = plt.title("Generalization error obtained by cross-validation")
legend(['Gen. Error','Lowest error'])

# plt.savefig('images/regress_Q2_V2_gen_error.pdf',bbox_inches = 'tight')
print(f"The mean optimal alpha leading to the lowest generalization error is:\n"
      f"{np.mean(best_alphas):.2f} +/- {np.std(best_alphas):.2f}")
print("POZO POUZIVAME ZDE POLYNOM, IKDYZ VYCHAZI LEPE, CHTEJI V REPORTU JEN CARU\PLANE/HYPERPLANE")

# print(y)
model_done = ridge.fit(X, y)
predict_first_bill_lenght = ridge.predict(X[0].reshape(1, -1))
print(predict_first_bill_lenght)  
#should be 39.1, is 40.08. Coz je chyba 2.25% coz sedi (protoze to presne tohodle penguina uz znalo, jinak je avg 5.5%)
# https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/