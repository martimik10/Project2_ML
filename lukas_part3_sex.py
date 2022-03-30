import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
from sklearn.metrics import classification_report
sns.set(style='whitegrid')
warnings.filterwarnings('ignore')

# READ DATA
print('\n\n\nREAD DATA\n')
data = pd.read_csv('dataset/penguins_original.csv')
print(data.head(5), end='\n\n')
print(data.describe())

# REMOVE NULL VALUES
print('\n\n\nREMOVE NULL VALUES\n')
print(data.isnull().any(), end='\n\n')
data = data.dropna(axis=0, how="any")
print(data.describe())

# ENCODE THE STRING VALUES
print('\n\n\nENCODE STRINGS\n')
le = LabelEncoder()
data['species_encoded'] = le.fit_transform(data['species'])
data['island_encoded'] = le.fit_transform(data['island'])
data['sex_encoded'] = le.fit_transform(data['sex'])
print(data.head(5))

# PLOT THE VALUES
plt.figure(figsize=(32, 10))
plt.subplot(2, 2, 1)
sns.distplot(data['bill_length_mm'])
plt.subplot(2, 2, 2)
sns.distplot(data['bill_depth_mm'])
plt.subplot(2, 2, 3)
sns.distplot(data['flipper_length_mm'])
plt.subplot(2, 2, 4)
sns.distplot(data['body_mass_g'])
plt.show()

# FIND BOUNDARY VALUES
print('\n\n\nFIND BOUDNARY VALUES\n')
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
            'body_mass_g', 'species_encoded', 'island_encoded', 'sex_encoded']
for column in data[features]:
    print("Highest allowed in {} is:{}".format(
        column, data[column].mean() + 3*data[column].std()))
    print("Lowest allowed in {} is:{}\n".format(
        column, data[column].mean() - 3*data[column].std()))

# FIND THE OUTLIERS (SPOILER: THERE ARE NONE)
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

# SELECT THE FEATURES (heatmap of correlation matrix with annotations in 2 different shades)
cor = data[features].corr()
hm1 = sns.heatmap(cor, annot=True, cmap='YlGnBu')
hm1.set(xlabel='Features', ylabel='Features',
        title="Correlation matrix of data\n")
plt.show()

# TEST AND TRAIN SET
print('\n\n\nTEST AND TRAIN SET\n')
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
            'body_mass_g', 'island_encoded', 'sex_encoded']
x = data[features]  # since these are the features we take them as x
# since species is the output or label we'll take it as y
y = data['species_encoded']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=12)
print("Shape of x_train:{}".format(x_train.shape))
print("Shape of x_test:{}".format(x_test.shape))
print("Shape of y_train:{}".format(y_train.shape))
print("Shape of y_test:{}".format(y_test.shape))

# SCALING
print('\n\n\nSCALING\n')
sc = StandardScaler()
scaled_x_train = sc.fit_transform(x_train)
scaled_x_test = sc.transform(x_test)
print(x_train)
print("", scaled_x_train)
x_train = scaled_x_train
x_test = scaled_x_test

# KNN
model1 = KNeighborsClassifier(n_neighbors=3)
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

# EVALUATE MODEL
print('\n\n\nEVALUATE MODEL\n')
ConfusionMatrix1 = confusion_matrix(y_test, y_prediction1)
print(ConfusionMatrix1)
ConfusionMatrix2 = confusion_matrix(y_test, y_prediction2)
print(ConfusionMatrix2)
ConfusionMatrix3 = confusion_matrix(y_test, y_prediction3)
print(ConfusionMatrix3)
ConfusionMatrix4 = confusion_matrix(y_test, y_prediction4)
print(ConfusionMatrix4)

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
plt.figure(figsize=(32, 10))
plt.subplot(2, 2, 1)
ax = sns.heatmap(ConfusionMatrix1, annot=True, cmap="YlGnBu")
ax.set_title('Confusion matrix')
ax.set_xlabel('Predicted values')
ax.set_ylabel('Actual values')


# DECISION TREE
plt.subplot(2, 2, 2)
ax = sns.heatmap(ConfusionMatrix2, annot=True, cmap="YlGnBu")
ax.set_title('Confusion matrix')
ax.set_xlabel('Predicted values')


# SVM
plt.subplot(2, 2, 3)
ax = sns.heatmap(ConfusionMatrix3, annot=True, cmap="YlGnBu")
ax.set_title('Confusion matrix')
ax.set_xlabel('Predicted values')

# LOGISTIC REGRESSION
plt.subplot(2, 2, 4)
ax = sns.heatmap(ConfusionMatrix4, annot=True, cmap="YlGnBu")
ax.set_title('Confusion matrix')
ax.set_xlabel('Predicted values')
plt.show()

# ROC CURVE KNN
plt.figure(figsize=(32, 10))
plt.subplot(2, 2, 1)
train_probs1 = model1.predict_proba(x_train)
train_probs1 = train_probs1[:, 1]
fpr1_train, tpr1_train, _ = roc_curve(y_train, train_probs1, pos_label=1)
test_probs1 = model1.predict_proba(x_test)
test_probs1 = test_probs1[:, 1]
fpr1_test, tpr1_test, _ = roc_curve(y_test, test_probs1, pos_label=1)
plt.plot(fpr1_train, tpr1_train, marker='.', label='train')
plt.plot(fpr1_test, tpr1_test, marker='.', label='validation')
plt.title('KNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

# ROC CURVE DECISION TREE
plt.subplot(2, 2, 2)
train_probs2 = model2.predict_proba(x_train)
train_probs2 = train_probs2[:, 1]
fpr2_train, tpr2_train, _ = roc_curve(y_train, train_probs2, pos_label=1)
test_probs2 = model2.predict_proba(x_test)
test_probs2 = test_probs2[:, 1]
fpr2_test, tpr2_test, _ = roc_curve(y_test, test_probs2, pos_label=1)
plt.plot(fpr2_train, tpr2_train, marker='.', label='train')
plt.plot(fpr2_test, tpr2_test, marker='.', label='validation')
plt.title('Decision Tree')
plt.xlabel('False Positive Rate')
plt.legend()

# ROC CURVE MODEL SVM
plt.subplot(2, 2, 3)
train_probs3 = model3.predict_proba(x_train)
train_probs3 = train_probs3[:, 1]
fpr3_train, tpr3_train, _ = roc_curve(y_train, train_probs3, pos_label=1)
test_probs3 = model3.predict_proba(x_test)
test_probs3 = test_probs3[:, 1]
fpr3_test, tpr3_test, _ = roc_curve(y_test, test_probs3, pos_label=1)
plt.plot(fpr3_train, tpr3_train, marker='.', label='train')
plt.plot(fpr3_test, tpr3_test, marker='.', label='validation')
plt.title('SVM')
plt.xlabel('False Positive Rate')
plt.legend()

# ROC CURVE LOGISTIC REGRESSION
plt.subplot(2, 2, 4)
train_probs4 = model4.predict_proba(x_train)
train_probs4 = train_probs4[:, 1]
fpr4_train, tpr4_train, _ = roc_curve(y_train, train_probs4, pos_label=1)
test_probs4 = model4.predict_proba(x_test)
test_probs4 = test_probs4[:, 1]
fpr4_test, tpr4_test, _ = roc_curve(y_test, test_probs4, pos_label=1)
plt.plot(fpr4_train, tpr4_train, marker='.', label='train')
plt.plot(fpr4_test, tpr4_test, marker='.', label='validation')
plt.title('Logistic Regression')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()
