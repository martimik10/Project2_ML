import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import mean
from numpy import absolute
from numpy import sqrt
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import sklearn.metrics
import plotly.express as px  # for plotting the scatter plot
import seaborn as sns  # For plotting the dataset in seaborn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import classification_report
import copy
sns.set(style='whitegrid')
warnings.filterwarnings('ignore')
from lukas_functions import *

# LOAD DATA
data = load_data('dataset/penguins_original.csv',disp=False)
plot_correlation(data)
plot_distribution(data)

# # TEST AND TRAIN SET
# # not suitable for k-fold algorithm
# x = data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
#           'body_mass_g', 'island_encoded', 'species_encoded']]
# y = data['sex_encoded']
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, random_state=42)
# print_size(x_train, y_train, x_test, y_test)
# sc = StandardScaler() #standardization of test and train set
# x_train = sc.fit_transform(x_train)
# x_test = sc.fit_transform(x_test)

# STANDARDIZATION
# no effect as the STANDARD SCALER ALSO STANDARDIZE THE DATA
temp = data[['bill_length_mm', 'bill_depth_mm',
             'flipper_length_mm', 'body_mass_g']]
data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
      'body_mass_g']] = (temp-temp.mean())/temp.std()

# STANDARD SCALER
data_x = data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
               'body_mass_g', 'gentoo',  'chinstrap', 'adelie',
               'torgersen', 'dream', 'biscoe','year']]
data_y = data['sex_encoded']
sc = StandardScaler()
data_x = sc.fit_transform(data_x)

# CROSS VALIDATION
kf = KFold(n_splits=10, shuffle=False)
gen_error = 0
index = 1
error_outloop = []
error_knn = []
error_decisiontree = []
error_svm = []
error_logisticregression = []
error_baseline = []
regression_par = []

decisiontree = DecisionTreeClassifier()  # Create models
svmmodel = svm.SVC(kernel='linear', probability=True)  # Linear Kernel
logisticregression = LogisticRegression()
knn = [KNeighborsClassifier(1), KNeighborsClassifier(2), KNeighborsClassifier(3), KNeighborsClassifier(4), KNeighborsClassifier(5), KNeighborsClassifier(6), KNeighborsClassifier(7), KNeighborsClassifier(8), KNeighborsClassifier(9), KNeighborsClassifier(10)]

print("INNER LOOP")
print("KNN - TREE - SVM - LOG.REG. - BASELINE")
for train_index, test_index in kf.split(data_x):
    print(index,end=' ')
    x_train, x_test, = data_x[train_index], data_x[test_index] 
    y_train, y_test = data_y[train_index], data_y[test_index]
    best_k = find_best_k(x_train, y_train, x_test, y_test)

    gen_errors_inner = cross_validation_inner(x_train, y_train, knn, best_k,decisiontree, svmmodel, logisticregression)
    models = [knn[best_k],decisiontree, svmmodel, logisticregression]
    best_modeltype_inner = models[np.argmin(gen_errors_inner)]
    best_model_inner = best_modeltype_inner.fit(x_train, y_train)
    y_prediction = best_model_inner.predict(x_test)
    gen_error += (len(test_index)/(len(data_x))) * (1-accuracy_score(y_test, y_prediction))
    error_outloop.append(1-accuracy_score(y_test, y_prediction))
    # in inner loop we choose the model based on argmin of generalization error
    # in outer loop we compute the generalization error once more for the best model

    # REGULARIZATION LOGISTIC REGRESSION
    lambda_interval = np.logspace(-3, 5, 200)
    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))
    for k in range(0, len(lambda_interval)):
        mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k])
        mdl.fit(x_train, y_train)
        y_train_est = mdl.predict(x_train).T
        y_test_est = mdl.predict(x_test).T
        train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)
        w_est = mdl.coef_[0]
        coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambda_interval[opt_lambda_idx]
    regression_par.append(10**np.round(np.log10(opt_lambda), 2))
    savefig_regularization(lambda_interval, train_error_rate,
                           test_error_rate, opt_lambda, index, min_error,coefficient_norm)
    index += 1
    
    
    
    # VOLUNTARY
    # just for double check how the other models behave
    # otherwise we choose only the best model from inner loop

    # KNN
    model1 = knn[best_k].fit(x_train, y_train)
    y_prediction1 = model1.predict(x_test)  # predict response
    error_knn.append(1-accuracy_score(y_test, y_prediction1))

    # DECISION TREE
    model2 = decisiontree.fit(x_train, y_train)
    y_prediction2 = model2.predict(x_test)  # predict response
    error_decisiontree.append(1-accuracy_score(y_test, y_prediction2))

    # SVM
    model3 = svmmodel.fit(x_train, y_train)
    y_prediction3 = model3.predict(x_test)  # predict response
    error_svm.append((1-accuracy_score(y_test, y_prediction3)))

    # LOGISTIC REGRESSION
    model4 = logisticregression.fit(x_train, y_train)
    y_prediction4 = model4.predict(x_test)  # predict response
    error_logisticregression.append((1-accuracy_score(y_test, y_prediction4)))

    # BASELINE
    y_prediction5 = np.ones(len(x_test), dtype=int) # 168x male, 165xfemale
    error_baseline.append((1-accuracy_score(y_test, y_prediction5)))

    # STATISTICAL TESTING
    alpha = 0.05
    theta1,theta2,theta3,theta4,theta5,theta6,theta7,theta8,theta9,theta10 = [],[],[],[],[],[],[],[],[],[]
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = [],[],[],[],[],[],[],[],[],[]
    CIL1,CIL2,CIL3,CIL4,CIL5,CIL6,CIL7,CIL8,CIL9,CIL10 = [],[],[],[],[],[],[],[],[],[]
    CIR1, CIR2, CIR3, CIR4, CIR5, CIR6, CIR7, CIR8, CIR9, CIR10 = [
    ], [], [], [], [], [], [], [], [], []

    stat_test = []
    #[thetahat, CI, p] = mcnemar(y_true, yhat1, yhat2, alpha=alpha)
    
    # KNN - Baseline 11
    theta, CIL, CIR, p = mcnemar(y_test, y_prediction1,y_prediction5, alpha=alpha)
    theta1.append(theta)
    CIL1.append(CIL)
    CIR1.append(CIR)
    p1.append(p)

    # TREE - Baseline 2
    theta, CIL, CIR, p = mcnemar(y_test, y_prediction2, y_prediction5, alpha=alpha)
    theta2.append(theta)
    CIL2.append(CIL)
    CIR2.append(CIR)
    p2.append(p)

    # SVM - Baseline 3
    theta, CIL, CIR, p = mcnemar(y_test, y_prediction3, y_prediction5, alpha=alpha)
    theta3.append(theta)
    CIL3.append(CIL)
    CIR3.append(CIR)
    p3.append(p)

    # LOG - Baseline 4
    theta, CIL, CIR, p = mcnemar(y_test, y_prediction4, y_prediction5, alpha=alpha)
    theta4.append(theta)
    CIL4.append(CIL)
    CIR4.append(CIR)
    p4.append(p)

    # # KNN - LOG 5
    theta, CIL, CIR, p = mcnemar(y_test, y_prediction1, y_prediction4, alpha=alpha)
    theta5.append(theta)
    CIL5.append(CIL)
    CIR5.append(CIR)
    p5.append(p)

    # # TREE - LOG 6
    theta, CIL, CIR, p = mcnemar(y_test, y_prediction2, y_prediction4, alpha=alpha)
    theta6.append(theta)
    CIL6.append(CIL)
    CIR6.append(CIR)
    p6.append(p)

    # # SVM - LOG 7
    theta, CIL, CIR, p = mcnemar(y_test, y_prediction3, y_prediction4, alpha=alpha)
    theta7.append(theta)
    CIL7.append(CIL)
    CIR7.append(CIR)
    p7.append(p)

    # # KNN - SVM 8
    theta, CIL, CIR, p = mcnemar(y_test, y_prediction1, y_prediction3, alpha=alpha)
    theta8.append(theta)
    CIL8.append(CIL)
    CIR8.append(CIR)
    p8.append(p)
    
    # # TREE - SVM 9
    theta, CIL, CIR, p = mcnemar(y_test, y_prediction2, y_prediction3, alpha=alpha)
    theta9.append(theta)
    CIL9.append(CIL)
    CIR9.append(CIR)
    p9.append(p)

    # # KNN - TREE 10
    theta, CIL, CIR, p = mcnemar(y_test, y_prediction1, y_prediction2, alpha=alpha)
    theta10.append(theta)
    CIL10.append(CIL)
    CIR10.append(CIR)
    p10.append(p)

print("\nOUTER LOOP")
print("Error Etest KNN:                 [%]", (100*np.array(error_knn)).round(2))
print("Error Etest Decision Tree:       [%]", (100*np.array(error_decisiontree)).round(2))
print("Error Etest SVM:                 [%]", (100*np.array(error_svm)).round(2))
print("Error Etest Logistic Regression: [%]", (100*np.array(error_logisticregression)).round(2))
print("Error Etest Baseline:            [%]", (100*np.array(error_baseline)).round(2),'\n')

print("STATISTICAL TESTING")

print("KNN - Baseline")
print("theta = theta_A-theta_B point estimate", np.mean(theta1),"\nCI: ", np.mean(CIL1), np.mean(CIR1), "\np-value", np.mean(p1))

print("\nTREE - Baseline")
print("theta = theta_A-theta_B point estimate", np.mean(theta1),
      "\nCI: ", np.mean(CIL2), np.mean(CIR2), "\np-value", np.mean(p2))

print("\nSVM - Baseline")
print("theta = theta_A-theta_B point estimate", np.mean(theta1),
      "\nCI: ", np.mean(CIL3), np.mean(CIR3), "\np-value", np.mean(p3))

print("\nLOG - Baseline")
print("theta = theta_A-theta_B point estimate", np.mean(theta1),
      "\nCI: ", np.mean(CIL4), np.mean(CIR4), "\np-value", np.mean(p4))

print("\nKNN - LOG")
print("theta = theta_A-theta_B point estimate", np.mean(theta1),
      "\nCI: ", np.mean(CIL5), np.mean(CIR5), "\np-value", np.mean(p5))

print("\nTREE - LOG")
print("theta = theta_A-theta_B point estimate", np.mean(theta1),
      "\nCI: ", np.mean(CIL6), np.mean(CIR6), "\np-value", np.mean(p6))

print("\nSVM - LOG")
print("theta = theta_A-theta_B point estimate", np.mean(theta1),
      "\nCI: ", np.mean(CIL7), np.mean(CIR7), "\np-value", np.mean(p7))

print("\nKNN - SVM")
print("theta = theta_A-theta_B point estimate", np.mean(theta1),
      "\nCI: ", np.mean(CIL8), np.mean(CIR8), "\np-value", np.mean(p8))

print("\nTREE - SVM")
print("theta = theta_A-theta_B point estimate", np.mean(theta1),
      "\nCI: ", np.mean(CIL9), np.mean(CIR9), "\np-value", np.mean(p9))

print("\nKNN - TREE")
print("theta = theta_A-theta_B point estimate", np.mean(theta1),
      "\nCI: ", np.mean(CIL10), np.mean(CIR10), "\np-value", np.mean(p10))

print("\nSUMMARY")
print("Error Inner Loop Best Model:     [%]", (100*np.array(error_outloop)).round(2))
print("Regression Parameter:            [-]", (np.array(regression_par)).round(2))
print("Total Generalization Error Estimate: ",100*gen_error.round(5),'%\n')



# # REPORT
# # print_report(y_test, y_prediction1, y_prediction2,
# #              y_prediction3, y_prediction4)
# # print_classification_report(y_test, y_prediction1,y_prediction2, y_prediction3, y_prediction4)
# plot_confusion_matrix(y_test1_best, y_test2_best, y_test3_best, y_test4_best, y_prediction1_best,
#                       y_prediction2_best, y_prediction3_best, y_prediction4_best)

# # ROC CURVE KNN
# plot_roc_curve(model1_best, model2_best, model3_best, model4_best,
#                x_train1_best, x_train2_best, x_train3_best, x_train4_best, x_test1_best, x_test2_best, x_test3_best, x_test4_best, y_train1_best, y_train2_best, y_train3_best, y_train4_best, y_test1_best, y_test2_best, y_test3_best, y_test4_best,)
