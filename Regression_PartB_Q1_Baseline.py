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
    model.fit(X, y)
    return model, inner_scores['test_r2'], outer_scores['test_r2']

model, inner, outer = nested_cv(X, y, 10, 10)
print(outer)

def predict_unseen(model, unseen_file, element=0):
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
    
    predict_unseen = model.predict(X_unseen)
    
    abs_true_error = np.round(abs(predict_unseen -  y_unseen_label), 2)
    perc_error = np.round(abs_true_error*100/y_unseen_label, 2)
    print("Predictions of unseen data:", np.round(predict_unseen, 2))
    print("True data:", y_unseen_label)
    print("Abs True error",abs_true_error )
    print("Error in %:", perc_error)
    print("Mean true error %", round(np.mean(perc_error), 2))
    return predict_unseen
    
    
#should be 39.1, is 40.08. Coz je chyba 2.25% coz sedi (protoze to presne tohodle penguina uz znalo, jinak je avg 5.5%)

################# STATISTICAL ANALYSIS ON UNSEEN DATA

unseen_file = "dataset/penguins_testing_regression_unseen.csv"
yhat2 = predict_unseen(model, unseen_file) 
#Further analysis in Stats file