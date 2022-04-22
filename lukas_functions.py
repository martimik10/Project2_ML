import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import mean
from numpy import absolute
from numpy import sqrt
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler, label_binarize, OneHotEncoder
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
import scipy
sns.set(style='whitegrid')
warnings.filterwarnings('ignore')


def load_data(path, disp):

    # READ DATA
    data = pd.read_csv(path)
    old_data = data.copy()

    # REMOVE NAN VALUES
    data = data.dropna(axis=0, how="any")
    data = data.reset_index(drop=True)

    # ENCODE THE STRING VALUES TO NUMBERS
    le = LabelEncoder()
    data['species_encoded'] = le.fit_transform(data['species'])
    data['island_encoded'] = le.fit_transform(data['island'])
    data['sex_encoded'] = le.fit_transform(data['sex'])

    # REMOVE OUTLIERS FROM BOUNDARY VALUES (SPOILER: THERE ARE NONE)
    # based on boxplot: https://towardsdev.com/applying-classification-algorithms-on-palmer-penguin-dataset-a41f6312b7f0
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

    # ONE HOT ENCODING
    encoder = OneHotEncoder(handle_unknown='ignore')
    features = ['island_encoded', 'species_encoded']
    encoder_data = pd.DataFrame(
        encoder.fit_transform(data[features]).toarray())
    data = data.join(encoder_data)
    data.columns.values[-6:] = ['gentoo',  'chinstrap', 'adelie',
                                'torgersen', 'dream', 'biscoe']
    #final_df.drop('sex_encoded', axis=1, inplace=True) deletes the unwanted columns

    # NO PRINTING
    if disp == False:
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
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
                'body_mass_g', 'species_encoded', 'island_encoded', 'sex_encoded']
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


def find_best_k(x_train, y_train, x_test, y_test):
    k_range = range(1, 10)
    error_rate = []
    for i in k_range: # optimazed for this dataset
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        pred_i = knn.predict(x_test)
        error_rate.append(np.mean(pred_i != y_test))
    plot_error_knn(error_rate,k_range)
    min_error = min(error_rate)
    best_k = np.argmin(error_rate)+k_range[0]-1
    print("Best K: ",best_k+1)
    return best_k

# CROSS VALIDATION INNTER LAYER
# finds the best model which we then train and test on a new dataset in outer loop

def cross_validation_inner(data_x, data_y, knn, best_k, decisiontree, svmmodel, logisticregression):
    kf = KFold(n_splits=10, shuffle=False)
    gen_error1, gen_error2, gen_error3, gen_error4, gen_error5 = 0, 0, 0, 0, 0
    data_y = data_y.reset_index(drop=True)

    for train_index, test_index in kf.split(data_x):
        x_train, x_test = data_x[train_index], data_x[test_index]
        y_test, y_train = data_y[test_index], data_y[train_index]

        # KNN
        model1 = knn[best_k].fit(x_train, y_train)
        y_prediction1 = model1.predict(x_test)  # predict response
        gen_error1 += (len(test_index)/(len(data_x))) * \
            (1-accuracy_score(y_test, y_prediction1))

        # DECISION TREE
        model2 = decisiontree.fit(x_train, y_train)
        y_prediction2 = model2.predict(x_test)  # predict response
        gen_error2 += (len(test_index)/(len(train_index))) * \
            (1-accuracy_score(y_test, y_prediction2))

        # SVM
        model3 = svmmodel.fit(x_train, y_train)
        y_prediction3 = model3.predict(x_test)  # predict response
        gen_error3 += (len(test_index)/(len(train_index))) * \
            (1-accuracy_score(y_test, y_prediction3))

        # LOGISTIC REGRESSION
        model4 = logisticregression.fit(x_train, y_train)
        y_prediction4 = model4.predict(x_test)  # predict response
        gen_error4 += (len(test_index)/(len(train_index))) * \
            (1-accuracy_score(y_test, y_prediction4))

        # BASELINE
        # 168x male, 165xfemale
        y_prediction5 = np.ones(len(x_test), dtype=int)
        gen_error5 += (len(test_index)/(len(train_index))) * \
            (1-accuracy_score(y_test, y_prediction5))

    gen_errors = [gen_error1, gen_error2, gen_error3, gen_error4, gen_error5]
    print(100*np.array(gen_errors).round(4)," [%]")
    return gen_errors #generalization error (calculated using validation err)


def plot_error_knn(error_rate, k_range):
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, error_rate, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.savefig('images/lukas/simple/Error Rate KNN.pdf')


def print_size(x_train, y_train, x_test, y_test):
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


def plot_confusion_matrix(y_test1, y_test2, y_test3, y_test4, y_prediction1, y_prediction2, y_prediction3, y_prediction4):
    ConfusionMatrix1 = confusion_matrix(y_test1, y_prediction1)
    ConfusionMatrix2 = confusion_matrix(y_test2, y_prediction2)
    ConfusionMatrix3 = confusion_matrix(y_test3, y_prediction3)
    ConfusionMatrix4 = confusion_matrix(y_test4, y_prediction4)

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


def plot_roc_curve(model1, model2, model3, model4, x_train1, x_train2, x_train3, x_train4, x_test1, x_test2, x_test3, x_test4, y_train1, y_train2, y_train3, y_train4, y_test1, y_test2, y_test3, y_test4):
    plt.subplots(2, 2, figsize=(10, 7))
    plt.suptitle('ROC Curve')
    plt.subplot(2, 2, 1)
    train_probs1 = model1.predict_proba(x_train1)
    train_probs1 = train_probs1[:, 1]
    fpr1_train, tpr1_train, _ = roc_curve(
        y_train1, train_probs1)  # pos_label=1
    test_probs1 = model1.predict_proba(x_test1)
    test_probs1 = test_probs1[:, 1]
    fpr1_test, tpr1_test, _ = roc_curve(
        y_test1, test_probs1)  # pos_label=1
    plt.plot(fpr1_train, tpr1_train, marker='.', label='train')
    plt.plot(fpr1_test, tpr1_test, marker='.', label='validation')
    plt.title('KNN')
    plt.ylabel('True Positive Rate')
    plt.legend()

    # ROC CURVE DECISION TREE
    plt.subplot(2, 2, 2)
    train_probs2 = model2.predict_proba(x_train2)
    train_probs2 = train_probs2[:, 1]
    fpr2_train, tpr2_train, _ = roc_curve(
        y_train2, train_probs2)  # pos_label=1
    test_probs2 = model2.predict_proba(x_test2)
    test_probs2 = test_probs2[:, 1]
    fpr2_test, tpr2_test, _ = roc_curve(
        y_test2, test_probs2)  # pos_label=1
    plt.plot(fpr2_train, tpr2_train, marker='.', label='train')
    plt.plot(fpr2_test, tpr2_test, marker='.', label='validation')
    plt.title('Decision Tree')
    plt.ylabel('True Positive Rate')
    plt.legend()

    # ROC CURVE MODEL SVM
    plt.subplot(2, 2, 3)
    train_probs3 = model3.predict_proba(x_train3)
    train_probs3 = train_probs3[:, 1]
    fpr3_train, tpr3_train, _ = roc_curve(
        y_train3, train_probs3)  # pos_label=1
    test_probs3 = model3.predict_proba(x_test3)
    test_probs3 = test_probs3[:, 1]
    fpr3_test, tpr3_test, _ = roc_curve(
        y_test3, test_probs3)  # pos_label=1
    plt.plot(fpr3_train, tpr3_train, marker='.', label='train')
    plt.plot(fpr3_test, tpr3_test, marker='.', label='validation')
    plt.title('SVM')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    # ROC CURVE LOGISTIC REGRESSION
    plt.subplot(2, 2, 4)
    train_probs4 = model4.predict_proba(x_train4)
    train_probs4 = train_probs4[:, 1]
    fpr4_train, tpr4_train, _ = roc_curve(
        y_train4, train_probs4)  # pos_label=1
    test_probs4 = model4.predict_proba(x_test4)
    test_probs4 = test_probs4[:, 1]
    fpr4_test, tpr4_test, _ = roc_curve(
        y_test4, test_probs4)  # pos_label=1
    plt.plot(fpr4_train, tpr4_train, marker='.', label='train')
    plt.plot(fpr4_test, tpr4_test, marker='.', label='validation')
    plt.title('Logistic Regression')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('images/lukas/simple/ROC Curve.pdf')


def savefig_regularization(lambda_interval, train_error_rate, test_error_rate, opt_lambda, index, min_error, coefficient_norm):
    plt.figure(figsize=(8, 8))
    plt.semilogx(lambda_interval, train_error_rate*100)
    plt.semilogx(lambda_interval, test_error_rate*100)
    plt.semilogx(opt_lambda, min_error*100, 'o')
    plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error*100, 2)
                                                   ) + ' % at 10^' + str(np.round(np.log10(opt_lambda), 2)))
    plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
    plt.ylabel('Error rate (%)')
    plt.title('Classification error')
    plt.legend(['Training error', 'Test error',
               'Test minimum'], loc='upper right')
    plt.grid()
    plt.savefig("images/lukas/regression/RegularizationErrorRate_"+str(index)+".pdf")
    plt.figure(figsize=(8, 8))
    plt.semilogx(lambda_interval, coefficient_norm, 'k')
    plt.ylabel('L2 Norm')
    plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
    plt.title('Parameter vector L2 norm')
    plt.grid()
    plt.savefig("images/lukas/regression/RegularizationL2_"+str(index)+".pdf")

def mcnemar(y_true, yhatA, yhatB, alpha=0.05):
    # perform McNemars test
    nn = np.zeros((2,2))
    c1 = yhatA - y_true == 0
    c2 = yhatB - y_true == 0
    nn[0,0] = sum(c1 & c2)
    nn[0,1] = sum(c1 & ~c2)
    nn[1,0] = sum(~c1 & c2)
    nn[1,1] = sum(~c1 & ~c2)
    n = sum(nn.flat)
    n12 = nn[0,1]
    n21 = nn[1,0]
    thetahat = (n12-n21)/n
    Etheta = thetahat
    Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)**2) )
    p = (Etheta + 1)*0.5 * (Q-1)
    q = (1-Etheta)*0.5 * (Q-1)
    CI = tuple(lm * 2 - 1 for lm in scipy.stats.beta.interval(1-alpha, a=p, b=q) )
    p = 2*scipy.stats.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)
    # print("Result of McNemars test using alpha=", alpha)
    # print("Comparison matrix n")
    # print(nn)
    # if n12+n21 <= 10:
    #     print("Warning, n12+n21 is low: n12+n21=",(n12+n21))
    # print("Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = ", CI)
    # print("p-value for two-sided test A and B have same accuracy (exact binomial test): p=", p)
    return [thetahat, CI[0],CI[1], p]