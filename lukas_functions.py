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


def load_data(path, disp):

    # READ DATA
    data = pd.read_csv(path)
    old_data = data.copy()

    # ENCODE THE STRING VALUES TO NUMBERS
    le = LabelEncoder()
    data['species_encoded'] = le.fit_transform(data['species'])
    data['island_encoded'] = le.fit_transform(data['island'])
    data['sex_encoded'] = le.fit_transform(data['sex'])

    # REMOVE NAN VALUES
    data = data.dropna(axis=0, how="any")

    # PLOT THE VALUES
    plt.subplots(2, 2, figsize=(10, 7))
    plt.suptitle('Distribution')
    plt.subplot(2, 2, 1)
    sns.distplot(data['bill_length_mm'])
    plt.subplot(2, 2, 2)
    sns.distplot(data['bill_depth_mm'])
    plt.subplot(2, 2, 3)
    sns.distplot(data['flipper_length_mm'])
    plt.subplot(2, 2, 4)
    sns.distplot(data['body_mass_g'])
    plt.savefig('images/lukas/Distribution.pdf')

    # REMOVE OUTLIERS FROM BOUNDARY VALUES (SPOILER: THERE ARE NONE)
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

    # FEATURE CORRELATION HEATMAP
    plt.figure(figsize=(15, 9))
    plt.title('Correlation matrix of data')
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
                'body_mass_g', 'species_encoded', 'island_encoded', 'sex_encoded']
    cor = data[features].corr()
    hm1 = sns.heatmap(cor, annot=True, cmap='YlGnBu')
    plt.savefig('images/lukas/Correlation.pdf')

    # NO PRINTING
    if disp==False:
        return data
    
    # DATA STATS
    print('\n\n\nDATA STATS\n')
    print('Old Data')
    print(old_data.head(5), end='\n\n')
    print("Old Data Stats")
    print(old_data.describe(), end='\n\n')
    print('Transformed Data')
    print(data.head(5), end='\n\n')
    print("Transformed Data Stats")
    print(data.describe())

    # REMOVE NAN VALAUES
    print('\n\n\nPARAMETERS INCLUDIGN ZERO VALUES \n')
    print(old_data.isnull().any())

    # OUTLIERS
    print('\n\n\nREMOVE OUTLIERS\n')
    for column in data[features]:
        print("Highest allowed in {} is:{}".format(
            column, data[column].mean() + 3*data[column].std()))
        print("Lowest allowed in {} is:{}".format(
            column, data[column].mean() - 3*data[column].std()))
    
    # RETURN VALUES
    print("\n\n")
    return data

def plot_correlation(data):
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
                'body_mass_g', 'species_encoded', 'island_encoded', 'sex_encoded']
    plt.figure(figsize=(15, 9))
    plt.title('Correlation matrix of data')
    cor = data[features].corr()
    hm1 = sns.heatmap(cor, annot=True, cmap='YlGnBu')
    plt.savefig('images/lukas/simple/Correlation.pdf')

def plot_distribution(data):
    plt.subplots(2, 2, figsize=(10, 7))
    plt.suptitle('Distribution')
    plt.subplot(2, 2, 1)
    sns.distplot(data['bill_length_mm'])
    plt.subplot(2, 2, 2)
    sns.distplot(data['bill_depth_mm'])
    plt.subplot(2, 2, 3)
    sns.distplot(data['flipper_length_mm'])
    plt.subplot(2, 2, 4)
    sns.distplot(data['body_mass_g'])
    plt.savefig('images/lukas/simple/Distribution.pdf')

def plot_error_knn(error_rate):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 10), error_rate, color='blue', linestyle='dashed', marker='o',
            markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.savefig('images/lukas/simple/Error Rate KNN.pdf')

def print_size(x_train, y_train, x_test,y_test):
    print('\n\n\nTEST AND TRAIN SET\n')
    print("Shape of x_train:{}".format(x_train.shape))
    print("Shape of x_test:{}".format(x_test.shape))
    print("Shape of y_train:{}".format(y_train.shape))
    print("Shape of y_test:{}".format(y_test.shape))


def print_report(y_test, y_prediction1, y_prediction2, y_prediction3, y_prediction4):
    report = pd.DataFrame()
    report['Actual values'] = y_test
    report['Predicted values KNN'] = y_prediction1
    report['Predicted values Decision tree'] = y_prediction2
    report['Predicted values SVM'] = y_prediction3
    report['Predicted values Logistic Regression'] = y_prediction4
    print('\n\n\nREPORT\n')
    print(report)


def print_classification_report(y_test, y_prediction1, y_prediction2, y_prediction3, y_prediction4):
    print('\n\n\nCLASSIFICATION REPORT\n')
    print("KNN")
    print(classification_report(y_test, y_prediction1))
    print("Decision tree")
    print(classification_report(y_test, y_prediction2))
    print("SVM")
    print(classification_report(y_test, y_prediction3))
    print("Logistic Regression")
    print(classification_report(y_test, y_prediction4))


def plot_confusion_matrix(y_test, y_prediction1, y_prediction2, y_prediction3, y_prediction4):
    ConfusionMatrix1 = confusion_matrix(y_test, y_prediction1)
    ConfusionMatrix2 = confusion_matrix(y_test, y_prediction2)
    ConfusionMatrix3 = confusion_matrix(y_test, y_prediction3)
    ConfusionMatrix4 = confusion_matrix(y_test, y_prediction4)

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
    plt.savefig('images/lukas/simple/Confusion Matrix.pdf')

def plot_roc_curve(model1, model2, model3, model4, x_train,x_test,y_train,y_test):
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
    plt.savefig('images/lukas/simple/ROC Curve.pdf')
