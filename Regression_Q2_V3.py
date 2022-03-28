import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn import metrics
from sklearn.metrics import mean_squared_error

from Regress_prep_loo import *

# X = pd.DataFrame(X)
# y = pd.DataFrame(y)
# X.info()
# y.info()

alphas = np.logspace(-3,4, num=1500)
cv = 10

#Two layer cross validation regularization
def nested_cross_validation(X=X, y=y, inner=10, outer=10, alphas=alphas):
    """
    return best alpha for each outer fold
    """
    model = Ridge()
    scoring ="neg_mean_squared_error"
    working = 1
    
    cv_outer = ShuffleSplit(n_splits=outer, test_size=0.5, random_state=0)
    cv_inner = ShuffleSplit(n_splits=inner, test_size=0.5, random_state=0)
    best_alphas = []
    best_errors = []
    for train_index, test_index in cv_outer.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        cv_scores = []
        for alpha in alphas:
            print("Working on fold", working, "/", len(alphas)*outer)
            model.set_params(alpha=alpha)
            scores = cross_validate(model, X_train, y_train, scoring=scoring, cv=cv_inner, return_train_score=False)
            cv_scores.append(scores["test_score"])
            working += 1
        cv_scores = np.array(cv_scores)
        best_alpha_index = np.argmax(np.mean(cv_scores, axis=0))
        best_error = np.max(np.mean(cv_scores, axis=0))
        best_alphas.append(alphas[best_alpha_index])
        best_errors.append(best_error)
    return best_alphas, best_errors

best_alphas, best_errors = nested_cross_validation() #return also inner folds
print(best_alphas)
# print(np.mean(np.multiply(-1, np.subtract(10,np.multiply(-1, best_errors)))))
print(np.mean(best_errors))



# def cross_validation(X=X, y=y, alphas=alphas, cv=cv):
#     """estimate generalization error"""
#     n_alphas = len(alphas)
#     error_train = np.zeros(n_alphas)
#     error_test = np.zeros(n_alphas)
#     for i, alpha in enumerate(alphas):
#         model = Ridge(alpha=alpha)
#         cv_error = cross_validate(model, X, y, cv=cv, scoring="neg_mean_squared_error", return_train_score=True)
#         error_train[i] = np.mean(cv_error["train_score"])
#         error_test[i] = np.mean(cv_error["test_score"])

 
#     return error_train, error_test

# error_train, error_test = cross_validation()
# print(-error_test, -error_train)

# def plot_error(error_test, alphas):
#     """plot generalization error"""
#     plt.plot(alphas, error_test)
#     plt.xlabel('alpha')
#     plt.ylabel('error')
#     plt.xscale('log')
#     plt.show()

# plot_error( error_test, alphas)





# w = list()
# for a in alphas:
#     ridge_clf = RidgeCV(alphas=[a],cv=10).fit(X, y)
#     w.append(ridge_clf.coef_)
# w = np.array(w)
# plt.semilogx((alphas),w)
# plt.title('Ridge coefficients as function of the regularization')
# plt.xlabel('Regularization factor')
# plt.ylabel('Mean Coefficient Values')
# plt.legend(attributeNames, prop={'size': 6})
# plt.savefig('images/regress_Q2_weigths.pdf',bbox_inches = 'tight')

# param = {
#     'alpha':np.logspace(-1, 5, num=150),
#     'fit_intercept':[True,False],
#     'normalize':[True,False],
# 'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
#        }

# #define model
# model = Ridge()
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# # define search
# search = GridSearchCV(model, param, scoring='r2', n_jobs=-1, cv=cv)

# # execute search
# result = search.fit(X, y)

# # summarize result
# print('Best Score: %s' % result.best_score_)
# print('Best Hyperparameters: %s' % result.best_params_)


