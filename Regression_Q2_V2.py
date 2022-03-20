y# -*- coding: utf-8 -*-
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
alphas = np.logspace(-2, 7, num=50)
# print(alphas)
ridge = make_pipeline(PolynomialFeatures(degree=5), StandardScaler(),
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

#By optimizing alpha, we see that the training and testing scores are close.
# It indicates that our model is not overfitting.

mse_alphas = [est[-1].cv_values_.mean(axis=0)
              for est in cv_results["estimator"]]
cv_alphas = pd.DataFrame(mse_alphas, columns=alphas)
# print("cv_alfas",cv_alphas)

plt.xscale("log")
# cv_alphas.mean(axis=0).plot(marker="+")
cv_alphas.mean(axis=0).plot(linewidth=3)

best_alphas = [est[-1].alpha_ for est in cv_results["estimator"]]
# print(best_alphas)

x = np.round(np.mean(best_alphas))
y = np.round(test_error.mean(), 2)
plt.text(x-3200, y+6, '({}, {})'.format(x, y))
plt.plot(x, y, marker="o", markersize=10, markeredgecolor="green", markerfacecolor="green")

plt.ylabel("Generalization error")
plt.xlabel("λ")
_ = plt.title("Generalization error obtained by cross-validation")
legend(['Gen. Error','Lowest error'])

# plt.savefig('images/regress_Q2_V2_gen_error.pdf',bbox_inches = 'tight')
print(f"The mean optimal alpha leading to the lowest generalization error is:\n"
      f"{np.mean(best_alphas):.2f} +/- {np.std(best_alphas):.2f}")

