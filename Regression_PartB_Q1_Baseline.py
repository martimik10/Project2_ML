from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
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

# linear_regression = make_pipeline(PolynomialFeatures(degree=1),
#                                   LinearRegression())
# cv_results = cross_validate(linear_regression, X, y,
#                             cv=10, scoring="neg_mean_squared_error",
#                             return_train_score=True,
#                             return_estimator=True)

# test_error = -cv_results["test_score"]
# print(test_error)

def nested_cv(X, y, inner, outer):
    """function for nested k-fold cross-validation of linear regression model"""
    model = LinearRegression()
    inner_cv = ShuffleSplit(n_splits=inner, test_size=0.2, random_state=0)
    outer_cv = ShuffleSplit(n_splits=outer, test_size=0.2, random_state=0)
    inner_scores = cross_validate(model, X, y, cv=inner_cv, scoring=['r2'], return_train_score=False)
    outer_scores = cross_validate(model, X, y, cv=outer_cv, scoring=['r2'], return_train_score=False)
    return inner_scores['test_r2'], outer_scores['test_r2']

inner, outer = nested_cv(X, y, 10, 10)
print(outer)