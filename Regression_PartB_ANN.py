# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import cross_validate
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import ShuffleSplit
# from sklearn import model_selection
# from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary, rocplot, confmatplot, rocplot, confmatplot
# from keras.models import Sequential
# from keras.layers import Activation, Dense
# import torch
# import numpy as np
# import pandas as pd
# import os
# from sklearn.model_selection import KFold, StratifiedKFold
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

from Regress_prep_loo import *

torch.cuda.is_available(﻿)
# # def neural_network(X, y, hidden_layers, kfold):
# #     """create neural network and cross validate on kfold to find best number of hidden layers"""
# #     model = Sequential()
# #     model.add(Dense(hidden_layers[0], activation='relu', input_dim=X.shape[1]))
# #     for i in range(len(hidden_layers)-1):
# #         model.add(Dense(hidden_layers[i+1], activation='relu'))
# #     model.add(Dense(1, activation='linear'))
# #     model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# #     cv_results = cross_validate(model, X, y, cv=kfold, scoring="neg_mean_squared_error", return_train_score=True)
# #     return cv_results


# # no_of_hidden_l = 2
# # hidden_layers = np.arange(1, no_of_hidden_l+1, 1)
# # print(hidden_layers)
# # neural_network(X, y, hidden_layers, kfold=10)

# from sklearn.model_selection import RepeatedKFold, cross_val_score
# from tensorflow.keras.models import * 
# from tensorflow.keras.layers import * 
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# def buildmodel():
#     model= Sequential([
#         Dense(10, activation="relu"),
#         Dense(5, activation="relu"),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mse', metrics=['mse'])
#     return(model)

# def build_ANN():
#     estimator= KerasRegressor(build_fn=buildmodel, epochs=500, batch_size=10, verbose=2)
#     results= cross_val_score(estimator, X, y, cv=10, n_jobs=1, verbose=2)  # 2 cpus
#     return results


# results = build_ANN()
# MSE = results.mean()  # Mean MSE
# print(results)

# Parameters for neural network classifier
n_hidden_units = 2      # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 1000

# K-fold crossvalidation
K = 3                   # only three folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
# Define the model
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

print('Training model of type:\n\n{}\n'.format(str(model())))
errors = [] # make a list for storing generalizaition error in each loop
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index])
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    errors.append(mse) # store error rate for current CV fold 
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')

# Display the MSE across folds
# summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
# summaries_axes[1].set_xlabel('Fold')
# summaries_axes[1].set_xticks(np.arange(1, K+1))
# summaries_axes[1].set_ylabel('MSE')
# summaries_axes[1].set_title('Test mean-squared-error')
    
print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,2]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# Print the average classification error rate
Error_each_fold = np.mean(errors, axis=1)
print("MSE errors: ", Error_each_fold)
print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))

# When dealing with regression outputs, a simple way of looking at the quality
# of predictions visually is by plotting the estimated value as a function of 
# the true/known value - these values should all be along a straight line "y=x", 
# and if the points are above the line, the model overestimates, whereas if the
# points are below the y=x line, then the model underestimates the value
plt.figure(figsize=(10,10))
y_est = y_test_est.data.numpy(); y_true = y_test.data.numpy()
axis_range = [np.min([np.squeeze(y_est), y_true])-1,np.max([np.squeeze(y_est), y_true])+1]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('Alcohol content: estimated versus true value (for last CV-fold)')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()

plt.show()

print('Ran Exercise 8.2.5')