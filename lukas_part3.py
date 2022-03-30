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

# READ DATA
print('\n\n\nREAD DATA\n')
data = pd.read_csv('dataset/penguins_original.csv')
old_data = data.copy()
print(data.head(5))

# ENCODE THE STRING VALUES TO NUMBERS
print('\n\n\nENCODE STRINGS\n')
le = LabelEncoder()
data['species_encoded'] = le.fit_transform(data['species'])
data['island_encoded'] = le.fit_transform(data['island'])
data['sex_encoded'] = le.fit_transform(data['sex'])
print(data.head(5))

# REMOVE NULL VALUES
print('\n\n\nREMOVE NULL VALUES\n')
#print(data.isnull().any(), end='\n\n')
data = data.dropna(axis=0, how="any")
print(data.head(5))

# PLOT THE VALUES


# REMOVE OUTLIERS FROM BOUNDARY VALUES (SPOILER: THERE ARE NONE)
print('\n\n\nREMOVE OUTLIERS\n')
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
            'body_mass_g', 'species_encoded', 'island_encoded', 'sex_encoded']
for column in data[features]:
    print("Highest allowed in {} is:{}".format(
        column, data[column].mean() + 3*data[column].std()))
    print("Lowest allowed in {} is:{}".format(
        column, data[column].mean() - 3*data[column].std()))
data[(data['bill_length_mm'] > 60.37587582718376) |
     (data['bill_length_mm'] < 27.61274692730728)]
data[(data['bill_depth_mm'] > 23.064207418059354) |
     (data['bill_depth_mm'] < 11.25675066577299)]
data[(data['flipper_length_mm'] > 243.0814956070838) |
     (data['flipper_length_mm'] < 158.94844451267664)]
data[(data['body_mass_g'] > 6623.565273989314) |
     (data['body_mass_g'] < 1794.548498465776)]
data[(data['species_encoded'] > 3.5932018846280505) |
     (data['species_encoded'] < -1.7488905073825416)]
data[(data['island_encoded'] > 2.79329552107842) | (
    data['island_encoded'] < -1.4938943234736295)]
data[(data['sex_encoded'] > 3.020135129128595) | (
    data['sex_encoded'] < -0.020135129128595164)]

# DATA STATS
print('\n\n\nDATA STATS\n')
print("Old Data")
print(old_data.describe(), end='\n\n')
print("Transformed Data")
print(data.describe())



# TEST AND TRAIN SET
print('\n\n\nTEST AND TRAIN SET\n')
x = data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
          'body_mass_g', 'island_encoded', 'species_encoded']]
y = data['sex_encoded']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2)  # to have the same results, we can choose one of the random_state=15 as argument
print("Shape of x_train:{}".format(x_train.shape))
print("Shape of x_test:{}".format(x_test.shape))
print("Shape of y_train:{}".format(y_train.shape))
print("Shape of y_test:{}".format(y_test.shape))

# SCALING / STANDARDIZATION
sc = StandardScaler()
scaled_x_train = sc.fit_transform(x_train)
scaled_x_test = sc.transform(x_test)
# print(x_train)
# print("", scaled_x_train)
x_train = scaled_x_train
x_test = scaled_x_test

# CROSS VALIDATION
print('\n\n\nCROSS VALIDATION')

knn = KNeighborsClassifier(n_neighbors=4)
CV_inner = model_selection.KFold(10, shuffle=True)
scores_train = cross_val_score(knn, x_train, y_train, cv=CV_inner)
scores_test = cross_val_score(knn, x_test, y_test, cv=CV)

RMSE_train = sqrt(mean(absolute(scores_train))) #root mean square
RMSE_test = sqrt(mean(absolute(scores_test))) #root mean square
print("KNN RMSE: ", RMSE_train, RMSE_test)

# SEARCHING FOR BEST K IN KNN
error_rate = []
for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.savefig('images/lukas/Error Rate KNN.pdf')

# REGULARIZATION LOGISTIC REGRESSION
lambda_interval = np.logspace(-8, 8, 50)
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
plt.figure(figsize=(8, 8))
plt.semilogx(lambda_interval, train_error_rate*100)
plt.semilogx(lambda_interval, test_error_rate*100)
plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error*100, 2)
                                               ) + ' % at 1e' + str(np.round(np.log10(opt_lambda), 2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error', 'Test error', 'Test minimum'], loc='upper right')
plt.grid()
plt.savefig("images/lukas/RegularizationErrorRate.pdf")

# PLOT REGULARIZATION L2
plt.figure(figsize=(8, 8))
plt.semilogx(lambda_interval, coefficient_norm, 'k')
plt.ylabel('L2 Norm')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.title('Parameter vector L2 norm')
plt.grid()
plt.savefig("images/lukas/RegularizationL2.pdf")

# KNN
model1 = KNeighborsClassifier(n_neighbors=4) # lowest error for k=4 
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
print('\n\n\nREPORT\n')
report = pd.DataFrame()
report['Actual values'] = y_test
report['Predicted values KNN'] = y_prediction1
report['Predicted values Decision tree'] = y_prediction2
report['Predicted values SVM'] = y_prediction3
report['Predicted values Logistic Regression'] = y_prediction4
print(report)
print(" KNN Accuracy Test Set: ", accuracy_score(y_test, y_prediction1))
print("Decission Tree Accuracy Test Set: ",
      accuracy_score(y_test, y_prediction2))
print(" SVM AccuracyTest Set: ", accuracy_score(y_test, y_prediction3))
print("Linear Regression Accuracy Test Set: ",
      accuracy_score(y_test, y_prediction4))

# EVALUATE MODEL
ConfusionMatrix1 = confusion_matrix(y_test, y_prediction1)
ConfusionMatrix2 = confusion_matrix(y_test, y_prediction2)
ConfusionMatrix3 = confusion_matrix(y_test, y_prediction3)
ConfusionMatrix4 = confusion_matrix(y_test, y_prediction4)
# print(ConfusionMatrix1) #print the 2D array

# CLASSIFICATION REPORT
print('\n\n\nCLASSIFICATION REPORT\n')
print("KNN")
print(classification_report(y_test, y_prediction1))
print("Decision tree")
print(classification_report(y_test, y_prediction2))
print("SVM")
print(classification_report(y_test, y_prediction3))
print("Logistic Regression")
print(classification_report(y_test, y_prediction4))

# KNN
plt.subplots(2, 2, figsize=(10, 7))
plt.suptitle('Confusion Matrix')
plt.subplot(2, 2, 1)
ax = sns.heatmap(ConfusionMatrix1, annot=True, cmap="YlGnBu")
plt.title("KNN")
ax.set_ylabel('Actual values')

# DECISION TREE
plt.subplot(2, 2, 2)
ax = sns.heatmap(ConfusionMatrix2, annot=True, cmap="YlGnBu")
plt.title("Decision Tree")
ax.set_ylabel('Actual values')

# SVM
plt.subplot(2, 2, 3)
ax = sns.heatmap(ConfusionMatrix3, annot=True, cmap="YlGnBu")
plt.title("SVM")
ax.set_xlabel('Predicted values')
ax.set_ylabel('Actual values')

# LOGISTIC REGRESSION
plt.subplot(2, 2, 4)
ax = sns.heatmap(ConfusionMatrix4, annot=True, cmap="YlGnBu")
plt.title("Logistic Regression")
ax.set_xlabel('Predicted values')
ax.set_ylabel('Actual values')
plt.savefig('images/lukas/Confusion Matrix.pdf')

# ROC CURVE KNN
plt.subplots(2, 2, figsize=(10, 7))
plt.suptitle('ROC Curve')
plt.subplot(2, 2, 1)
train_probs1 = model1.predict_proba(x_train)
train_probs1 = train_probs1[:, 1]
fpr1_train, tpr1_train, _ = roc_curve(y_train, train_probs1)  # pos_label=1
test_probs1 = model1.predict_proba(x_test)
test_probs1 = test_probs1[:, 1]
fpr1_test, tpr1_test, _ = roc_curve(
    y_test, test_probs1)  # pos_label=1
plt.plot(fpr1_train, tpr1_train, marker='.', label='train')
plt.plot(fpr1_test, tpr1_test, marker='.', label='validation')
plt.title('KNN')
plt.ylabel('True Positive Rate')
plt.legend()

# ROC CURVE DECISION TREE
plt.subplot(2, 2, 2)
train_probs2 = model2.predict_proba(x_train)
train_probs2 = train_probs2[:, 1]
fpr2_train, tpr2_train, _ = roc_curve(y_train, train_probs2)  # pos_label=1
test_probs2 = model2.predict_proba(x_test)
test_probs2 = test_probs2[:, 1]
fpr2_test, tpr2_test, _ = roc_curve(
    y_test, test_probs2)  # pos_label=1
plt.plot(fpr2_train, tpr2_train, marker='.', label='train')
plt.plot(fpr2_test, tpr2_test, marker='.', label='validation')
plt.title('Decision Tree')
plt.ylabel('True Positive Rate')
plt.legend()

# ROC CURVE MODEL SVM
plt.subplot(2, 2, 3)
train_probs3 = model3.predict_proba(x_train)
train_probs3 = train_probs3[:, 1]
fpr3_train, tpr3_train, _ = roc_curve(y_train, train_probs3)  # pos_label=1
test_probs3 = model3.predict_proba(x_test)
test_probs3 = test_probs3[:, 1]
fpr3_test, tpr3_test, _ = roc_curve(
    y_test, test_probs3)  # pos_label=1
plt.plot(fpr3_train, tpr3_train, marker='.', label='train')
plt.plot(fpr3_test, tpr3_test, marker='.', label='validation')
plt.title('SVM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

# ROC CURVE LOGISTIC REGRESSION
plt.subplot(2, 2, 4)
train_probs4 = model4.predict_proba(x_train)
train_probs4 = train_probs4[:, 1]
fpr4_train, tpr4_train, _ = roc_curve(y_train, train_probs4)  # pos_label=1
test_probs4 = model4.predict_proba(x_test)
test_probs4 = test_probs4[:, 1]
fpr4_test, tpr4_test, _ = roc_curve(
    y_test, test_probs4)  # pos_label=1
plt.plot(fpr4_train, tpr4_train, marker='.', label='train')
plt.plot(fpr4_test, tpr4_test, marker='.', label='validation')
plt.title('Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('images/lukas/ROC Curve.pdf')
