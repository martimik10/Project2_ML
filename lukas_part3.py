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
data = pd.read_csv('dataset/penguins_original.csv')
print(data.head(5))
print(data.describe())

# REMOVING NULL VALUES
print(data.isnull().any())
data = data.dropna(axis=0, how="any")
print(data.describe())

# ENCODE THE STRING VALUES
le = LabelEncoder()
data['species_encoded'] = le.fit_transform(data['species'])
data['island_encoded'] = le.fit_transform(data['island'])
data['sex_encoded'] = le.fit_transform(data['sex'])
print(data.head(5))

# PLOT THE VALUES
plt.figure(figsize=(32,10))
plt.subplot(2,2,1)
sns.distplot(data['bill_length_mm'])
plt.subplot(2,2,2)
sns.distplot(data['bill_depth_mm'])
plt.subplot(2,2,3)
sns.distplot(data['flipper_length_mm'])
plt.subplot(2,2,4)
sns.distplot(data['body_mass_g'])
plt.show()

# FIND BOUNDARY VALUES
features=['bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g', 'species_encoded','island_encoded', 'sex_encoded']
for column in data[features]:
    print("Highest allowed in {} is:{}".format(column,data[column].mean() + 3*data[column].std()))
    print("Lowest allowed in {} is:{}\n\n".format(column,data[column].mean() - 3*data[column].std()))
    
# FIND THE OUTLIERS (SPOILER: THERE ARE NONE)
data[(data['bill_length_mm'] > 60.37587582718376) | (data['bill_length_mm'] < 27.61274692730728)]
data[(data['bill_depth_mm'] > 23.064207418059354) | (data['bill_depth_mm'] < 11.25675066577299)]
data[(data['flipper_length_mm'] > 243.0814956070838) | (data['flipper_length_mm'] < 158.94844451267664)]
data[(data['body_mass_g'] > 6623.565273989314) | (data['body_mass_g'] < 1794.548498465776)]
data[(data['species_encoded'] > 3.5932018846280505) | (data['species_encoded'] < -1.7488905073825416)]
data[(data['island_encoded'] > 2.79329552107842) | (data['island_encoded'] < -1.4938943234736295)]
data[(data['sex_encoded'] > 3.020135129128595) | (data['sex_encoded'] < -0.020135129128595164)]

# SELECT THE FEATURES (heatmap of correlation matrix with annotations in 2 different shades)
cor=data[features].corr()
hm1 = sns.heatmap(cor, annot = True,cmap='YlGnBu')
hm1.set(xlabel='\nFeatures', ylabel='Features\t', title = "Correlation matrix of data\n")
plt.show()

# SPLIT INTO TEST AND TRAIN SET
features=['bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g','island_encoded', 'sex_encoded']
x=data[features]# since these are the features we take them as x
y=data['species_encoded']# since species is the output or label we'll take it as y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=12)
print("\nShape of x_train:\n{}".format(x_train.shape))
print("\nShape of x_test:\n{}".format(x_test.shape))
print("\nShape of y_train:\n{}".format(y_train.shape))
print("\nShape of y_test:\n{}".format(y_test.shape))

# SCALING
sc = StandardScaler()
scaled_x_train = sc.fit_transform(x_train)
scaled_x_test = sc.transform(x_test)
print(x_train)
print("____________________________________________________________________________")
print("",scaled_x_train)

# KNN
x_train=scaled_x_train
x_test=scaled_x_test
model1=KNeighborsClassifier(n_neighbors=3)
model1.fit(x_train, y_train)
y_prediction1= model1.predict(x_test) # predict response
report=pd.DataFrame()
report['Actual values']=y_test
report['Predicted values KNN']= y_prediction1
print(report)

# DECISION TREE
model2= DecisionTreeClassifier()
model2 = model2.fit(x_train,y_train)
y_prediction2 = model2.predict(x_test) # predict response
report['Predicted values Decision tree']= y_prediction2
print(report)

# SVM
model3= svm.SVC(kernel='linear',probability=True) # Linear Kernel
model3.fit(x_train, y_train)
y_prediction3 = model3.predict(x_test) # predict response
report['Predicted values SVM']= y_prediction3
print(report)

# EVALUATE MODEL
ConfusionMatrix1=confusion_matrix(y_test,y_prediction1)
print(ConfusionMatrix1)
ConfusionMatrix2=confusion_matrix(y_test,y_prediction2)
print(ConfusionMatrix2)
ConfusionMatrix3=confusion_matrix(y_test,y_prediction3)
print(ConfusionMatrix3)

# KNN
ax=sns.heatmap(ConfusionMatrix1,annot=True,cmap="YlGnBu")
ax.set_title('Confusion matrix')
ax.set_xlabel('Predicted values')
ax.set_ylabel('Actual values')
plt.show()

# DECISION TREE
ax=sns.heatmap(ConfusionMatrix2,annot=True,cmap="YlGnBu")
ax.set_title('Confusion matrix')
ax.set_xlabel('Predicted values')
ax.set_ylabel('Actual values')
plt.show()

# SVM
ax=sns.heatmap(ConfusionMatrix3,annot=True,cmap="YlGnBu")
ax.set_title('Confusion matrix')
ax.set_xlabel('Predicted values')
ax.set_ylabel('Actual values')
plt.show()

# ROC CURVE AND SCORE

fpr1, tpr1, thresh1 = roc_curve(y_test, y_prediction1, pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, y_prediction2, pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(y_test, y_prediction3, pos_label=1)

# ROC CURVE FOR TPR = FPR
train_pred_prob1 = model1.predict_proba(x_train)
#y_pred_prob1 = y_pred_prob1[:, 1]
#roc_auc_score(y_test, y_pred_prob1, multi_class='ovo', average='weighted')
#roc_auc_score(y_test, y_pred_prob2, multi_class='ovo', average='weighted')
#roc_auc_score(y_test, y_pred_prob3, multi_class='ovo', average='weighted')
fpr1, tpr1, thresh1 = roc_curve(y_train, train_pred_prob1[:,1], pos_label=1)
#fpr2, tpr2, thresh2 = roc_curve(y_test, y_pred_prob2[:,1], pos_label=1)
#fpr3, tpr3, thresh3 = roc_curve(y_test, y_pred_prob3[:,1], pos_label=1)
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
plt.plot(fpr1, tpr1, linestyle='--', color='black', label='KNN')
plt.plot(fpr2, tpr2, linestyle='--', color='green', label='Descision Tree')
plt.plot(fpr3, tpr3, linestyle='--', color='red', label='SVM')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
#plt.savefig('ROC', dpi=300)
plt.show()

# CLASSIFICATION REPORT
print("KNN")
print(classification_report(y_test, y_prediction1))
print("Decision tree")
print(classification_report(y_test, y_prediction2))
print("SVM")
print(classification_report(y_test, y_prediction3))

#######################################################

# LOGISTIC REGRESSION
model = LogisticRegression().fit(x_train, y_train)
pred = model.predict(x_test)

# EVAL
print('CONFUSION MATRIX')
print(confusion_matrix(y_test, pred))
print('CLASSIFICATION REPORT\n')
print(classification_report(y_test, pred))

# ROC CURVE
print('ROC CURVE')
train_probs = model.predict_proba(x_train)
train_probs1 = train_probs[:, 1]
fpr0, tpr0, thresholds0 = roc_curve(y_train, train_probs1, pos_label=1)

test_probs = model.predict_proba(x_test)
test_probs1 = test_probs[:, 1]
fpr1, tpr1, thresholds1 = roc_curve(y_test, test_probs1, pos_label=1)

plt.plot(fpr0, tpr0, marker='.', label='train')
plt.plot(fpr1, tpr1, marker='.', label='validation')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

