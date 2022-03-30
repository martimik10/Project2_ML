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
import statsmodels.api as sm
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

# TEST AND TRAIN SET
x = data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
          'body_mass_g', 'island_encoded', 'species_encoded']]
y = data['sex_encoded']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
#print_size(x_train, y_train, x_test, y_test)

# STANDARDIZATION
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# SEARCHING FOR BEST K IN KNN
error_rate = []
for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))
plot_error_knn(error_rate)
min_error = min(error_rate)
best_k = np.argmin(error_rate)+1
print("\nKNN Parameter")
print("Best K: ",best_k)

# KNN
model1 = KNeighborsClassifier(n_neighbors=4)  # lowest error for k=4
model1 = model1.fit(x_train, y_train)
y_prediction1 = model1.predict(x_test)  # predict response

# DECISION TREE
model2 = DecisionTreeClassifier()
model2 = model2.fit(x_train, y_train)
y_prediction2 = model2.predict(x_test)  # predict response

# SVM
model3 = svm.SVC(kernel='linear', probability=True)  # Linear Kernel
model3 = model3.fit(x_train, y_train)
y_prediction3 = model3.predict(x_test)  # predict response

# LOGISTIC REGRESSION
model4 = LogisticRegression()
model4 = model4.fit(x_train, y_train)
y_prediction4 = model4.predict(x_test)  # predict response

# REPORT
# print_report(y_test, y_prediction1, y_prediction2,
#              y_prediction3, y_prediction4)
# print_classification_report(y_test, y_prediction1,y_prediction2, y_prediction3, y_prediction4)
plot_confusion_matrix(y_test, y_prediction1, y_prediction2,
                      y_prediction3, y_prediction4)

# ACCURACY
print("\n\nACCURACY")
print("KNN: ", accuracy_score(y_test, y_prediction1))
print("Decission Tree: ",
      accuracy_score(y_test, y_prediction2))
print("SVM: ", accuracy_score(y_test, y_prediction3))
print("Logistic Regression: ",
      accuracy_score(y_test, y_prediction4),"\n\n")

# ROC CURVE KNN
plot_roc_curve(model1, model2, model3, model4, x_train, x_test, y_train, y_test)
