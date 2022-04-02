import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
from torch.autograd import Variable

# from Regress_prep_ann import *
from Regress_prep_loo import *

# for n_hidden_units in range(start, hidden_units_array+1):

def nested_CV_ANN(X, y):
    outer_folds = 3
    CV_out = model_selection.KFold(outer_folds, shuffle=True)
    k_out=0 #numbering
    outer_h = []
    outer_mse = []
    for train_index_out, test_index_out in CV_out.split(X,y):    
        X_train_out = X[train_index_out]
        y_train_out = y[train_index_out]
        X_test_out = X[test_index_out]
        y_test_out = y[test_index_out]
        
        
        start = 1
        stop = 10
        h_errors = []
        for i in range(start, stop+1):
            n_hidden_units = i
            #best range by brute force is around 101 layers
            
            n_replicates = 1        # number of networks trained in each k-fold
            max_iter = 10000
            
            # K-fold crossvalidation
            K = 3                   # only three folds to speed up this example
            CV = model_selection.KFold(K, shuffle=True)
            
            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss:
                
            
            print('Training model of type:\n\n{}\n'.format(str(model())))
            errors = [] # make a list for storing generalizaition error in each loop
            for (k, (train_index, test_index)) in enumerate(CV.split(X_train_out,y_train_out)): 
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
                
                weights = [net[i].weight.data.numpy().T for i in [0,2]]
                biases = [net[i].bias.data.numpy() for i in [0,2]]
                tf =  [str(net[i]) for i in [1,2]]
                draw_neural_net(weights, biases, tf, attribute_names=attributeNames)
            
            # print(np.sqrt(np.mean(errors))) #MSE
            h_errors.append(np.sqrt(np.mean(errors)))
              
        print(h_errors)
        print("best H:", np.argmin(h_errors)+1)
        outer_h.append(np.argmin(h_errors)+1)
        #   model, train_loss, train_accuracy, valid_loss, valid_accuracy, epoch, lr = train_neural_net(model, loss_func, optimizer, lr_scheduler, X, Y, X_valid=X_valid, Y_valid=Y_valid, validate_every=validate_every, validate_loss_func=validate_loss_func, validate_model=validate_model, validate_scheduler=validate_scheduler, validate_every_scheduler=validate_every_scheduler)
        # train_losses.append(train_loss)
        # train_accuracies.append(train_accuracy)
        # valid_losses.append(valid_loss)
        # valid_accuracies.append(valid_accuracy)
        outer_mse.append(np.min(h_errors))
        k_out += 1

    print("allbest Hs", outer_h)
    print("all best errs", outer_mse)

# nested_CV_ANN(X, y)

#############################################
#Train on best number of hidden layers for statistics

def train_on_best(X, y):
    n_hidden_units = 6 # chosen in CV
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 100000
    K = 10                   # only three folds to speed up this example
    CV = model_selection.KFold(K, shuffle=True)
    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss:
        
    
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
    

    return net

model = train_on_best(X, y)
print("Model", model)

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
    
    new_x = Variable(torch.Tensor(X_unseen))
    y_pred = model(new_x)
    y_pred = y_pred.data.numpy().squeeze()
    
    return y_pred

unseen_file = "dataset/penguins_testing_regression_unseen.csv"
yhat3 = predict_unseen(model, unseen_file) 
print("Predicted:", yhat3)