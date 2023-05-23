import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib import style

data = np.genfromtxt("train.csv", delimiter=",")
X_org = data[:, 2:]
y_org = data[:, 1]
Y_tr = np.reshape(y_org, (len(y_org), 1)) 
X_tr = np.c_[np.ones((len(X_org), 1)), X_org]
data = np.genfromtxt("validation.csv", delimiter=",")
X_org_val = data[:, 2:]
y_org_val = data[:, 1]
Y_val = np.reshape(y_org_val, (len(y_org_val), 1)) 
X_val = np.c_[np.ones((len(X_org_val), 1)), X_org_val]


def Mean_Squared_Error(X, y, theta):
    return np.square(X.dot(theta) - y).mean()
def Mean_Absolute_Error(X, y, theta):
    return np.abs(X.dot(theta) - y).mean()
def CostFun(X, y, theta):
    return Mean_Squared_Error(X, y, theta)/2

def theta_initial(X):
    theta = np.zeros([len(X[0]), 1])
    return theta

def gradient_descent_maxit(X_tr,Y_tr,rate, max_iter, X_val, Y_val):
    
    Theta = theta_initial(X_tr)
    bestTheta = Theta
    minimum_mse = 1
    N = len(X_tr)
    trainLoss = []
    validationLoss = []
    iter = []
    for i in range(max_iter):
        gradients = 2/N * X_tr.T.dot(X_tr.dot(Theta) - Y_tr)
        Theta = Theta - rate * gradients
        curr_mse = Mean_Squared_Error(X_val, Y_val, Theta)
        if curr_mse < minimum_mse:
            minimum_mse = curr_mse
            bestTheta = Theta
        validationLoss.append(curr_mse)
        iter.append(i)
        trainLoss.append(Mean_Squared_Error(X_tr, Y_tr, Theta))

    plt.plot(iter,trainLoss, color="r")
    plt.show()
    plt.plot(iter,validationLoss,color="r")
    plt.show()
    return bestTheta, trainLoss, validationLoss

def ridge_gradient_descent_maxit(X_tr,Y_tr,rate, lambda_value, max_iter, X_val, Y_val):
    theta = theta_initial(X_tr)
    bestTheta = theta
    minimum_mse = 100
    N = len(X_tr)
    trainLoss = []
    validationLoss = []
    iter = []
    for i in range(max_iter):
        gradients = np.add((2/N * (X_tr.T.dot(X_tr.dot(theta) - Y_tr))), ((2*lambda_value)/N)*(theta.T))
        theta = theta - rate * gradients
        curr_mse = Mean_Squared_Error(X_val, Y_val, theta)
        if curr_mse < minimum_mse:
            minimum_mse = curr_mse
            bestTheta = theta
        validationLoss.append(curr_mse)
        trainLoss.append(Mean_Squared_Error(X_tr, Y_tr, theta))
        iter.append(i)
    plt.plot(iter,trainLoss,color="r")
    plt.show()
    plt.plot(iter,validationLoss,color="r")
    plt.show()
    return bestTheta, trainLoss, validationLoss

def RelDec(X_tr, Y_tr, theta0, theta1):
    return (CostFun(X_tr, Y_tr, theta1) - CostFun(X_tr, Y_tr, theta0))/CostFun(X_tr, Y_tr, theta0)

def gradient_descent(X,y,learningrate, threshold, X_val, y_val):
    theta = theta_initial(X)
    N = len(X)
    rel_cost = -sys.maxsize
    i = 0
    trainLoss = []
    validationLoss = []
    iterr = []
    while rel_cost < threshold and i < 1000:
        gradients = 2/N * X.T.dot(X.dot(theta) - y)
        theta_new = theta - learningrate * gradients
        rel_cost = RelDec(X_val, y_val, theta, theta_new)
        trainLoss.append(Mean_Squared_Error(X_tr, Y_tr, theta_new))
        validationLoss.append(Mean_Squared_Error(X_val, Y_val, theta_new))
        theta = theta_new
        i = i + 1
        iterr.append(i)
    plt.plot(iterr,trainLoss, color="r")
    plt.show()
    plt.plot(iterr,validationLoss,color="r")
    plt.show()
    return theta, trainLoss, validationLoss

def ridge_gradient_descent(X_tr,Y_tr,rate, lambda_value, threshold, X_val, Y_val):
    theta = theta_initial(X_tr)
    N = len(X_tr)
    rel_cost = -sys.maxsize
    i = 0
    trainLoss = []
    validationLoss = []
    iter = []
    while rel_cost < threshold and i < 1000:
        gradients = (2/N * (X_tr.T.dot(X_tr.dot(theta) - Y_tr))) + ((2*lambda_value)/N)*(theta.T)
        theta_new = theta - rate * gradients
        rel_cost = RelDec(X_val, Y_val, theta, theta_new)
        trainLoss.append(Mean_Squared_Error(X_tr, Y_tr, theta_new))
        validationLoss.append(Mean_Squared_Error(X_val, Y_val, theta_new))
        theta = theta_new
        iter.append(i)
        i = i + 1
    plt.plot(iter, trainLoss,color="r")
    plt.show()
    plt.plot(iter,validationLoss,color="r")
    plt.show()
    return theta, trainLoss, validationLoss

def linear_regression():

    rate = 0.001
    max_iter= 1000
    threshold = -0.0001

    weights, t_lst, v_lst = gradient_descent(X_tr, Y_tr, rate, threshold, X_val, Y_val)
    print("If we use threshold and keep the learning rate = " , rate  )
    print("MSE of training data: ", Mean_Squared_Error(X_tr, Y_tr, weights) )
    print("MAE of training data: ", Mean_Absolute_Error(X_tr, Y_tr, weights) )
    print("MSE of validation data: ", Mean_Squared_Error(X_val, Y_val, weights) )
    print("MAE of validation data: ", Mean_Absolute_Error(X_val, Y_val, weights) )

    weights, t_lst, v_lst = gradient_descent_maxit(X_tr, Y_tr, rate, max_iter, X_val, Y_val)
    print("If we don't use threshold and keep the learning rate = " , rate  )
    print("MSE of training data: ", Mean_Squared_Error(X_tr, Y_tr, weights) )
    print("MAE of training data: ", Mean_Absolute_Error(X_tr, Y_tr, weights) )
    print("MSE of validation data: ", Mean_Squared_Error(X_val, Y_val, weights) )
    print("MAE of validation data: ", Mean_Absolute_Error(X_val, Y_val, weights) )


    f = open("demofile.txt", "w")
    for i in weights:
        f.write(str(i[0]))
        f.write("\n")
    f.close()

def ridge_regression():

    rate = 0.001
    threshold = -0.0001
    max_iter= 1000
    lambda_value = 5

    weights, t_lst, v_lst = ridge_gradient_descent(X_tr, Y_tr, rate, lambda_value, threshold, X_val, Y_val)

    print("If we use threshold and keep the learning rate = " , rate  )
    print("MSE of training data: ", Mean_Squared_Error(X_tr, Y_tr, weights) )
    print("MAE of training data: ", Mean_Absolute_Error(X_tr, Y_tr, weights) )
    print("MSE of validation data: ", Mean_Squared_Error(X_val, Y_val, weights) )
    print("MAE of validation data: ", Mean_Absolute_Error(X_val, Y_val, weights) )

    weights, t_lst, v_lst = ridge_gradient_descent_maxit(X_tr, Y_tr, rate, lambda_value, max_iter, X_val, Y_val)
    print("If we don't use threshold and keep the learning rate = " , rate  )
    print("MSE of training data: ", Mean_Squared_Error(X_tr, Y_tr, weights) )
    print("MAE of training data: ", Mean_Absolute_Error(X_tr, Y_tr, weights) )
    print("MSE of validation data: ", Mean_Squared_Error(X_val, Y_val, weights) )
    print("MAE of validation data: ", Mean_Absolute_Error(X_val, Y_val, weights) )

    # print(t_lst)
    # print(v_lst)

    # f = open("demofile.txt", "w")
    # for i in weights:
    #     f.write(str(i[0]))
    #     f.write("\n")
    # f.close()

import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
def sklLibrary():

    ml = LinearRegression()
    ml.fit(X_tr, Y_tr)
    pred = ml.predict(X_val)
    # print(pred)
    # print(ml.intercept_)
    # print(ml.coef_)

    print("MSE = " ,mean_squared_error(Y_val, pred)) #0.8178311302215785
    print("MAE = " ,mean_absolute_error(Y_val, pred)) #0.7116702931834721

def bestfeatures(n):
    # 10 best features

    from sklearn.feature_selection import SelectKBest, chi2, f_regression
    X_trnew =  SelectKBest(score_func=chi2,k=n).fit(X_tr, Y_tr)
    X_trnew.get_support()
    X_trnew.transform(X_tr)
    # print(X_trnew.transform(X_val))

    lm = LinearRegression()
    lm.fit(X_trnew.transform(X_tr), Y_tr)

    prednew = lm.predict(X_trnew.transform(X_val))
    print("number of features is " , n)
    print("MSE new is = " ,mean_squared_error(Y_val, prednew)) 
    print("MAE new is = " ,mean_absolute_error(Y_val, prednew)) 
    print("==========================================================================")
    return (mean_squared_error(Y_val, prednew))
    print("/n SelectFromModel")

    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import Ridge

    clf = Ridge(alpha=0.1, tol=0.001)
    clf.fit(X_tr, Y_tr)
    model = SelectFromModel(clf,max_features=10,prefit=True)
    new = model.transform(X_val)
    # print(new.shape) # 10
    predridge = Ridge.predict(clf, X_val)
    # print(predridge)
    print("MSE after using ridge is = " ,mean_squared_error(Y_val, predridge)) 
    print("MAE after using ridge is = " ,mean_absolute_error(Y_val, predridge)) 

def normalize(x):
    mean = np.mean(x)
    stanDev = np.std(x)
    #print(mean, stanDev)
    for i in range(len(x)):
        x[i] = (x[i] - mean) / stanDev
    return x

def visualisation_normal():

    rate = 0.001
    max_iter= 1000
    threshold = -0.0001

    weights, t_lst, v_lst = gradient_descent(normalize(X_tr), Y_tr, rate, threshold, X_val, Y_val)
    print("If we use threshold and keep the learning rate = " , rate  )
    print("MSE of training data: ", Mean_Squared_Error(normalize(X_tr), Y_tr, weights) )
    print("MAE of training data: ", Mean_Absolute_Error(normalize(X_tr), Y_tr, weights) )
    print("MSE of validation data: ", Mean_Squared_Error(X_val, Y_val, weights) )
    print("MAE of validation data: ", Mean_Absolute_Error(X_val, Y_val, weights) )

    weights, t_lst, v_lst = gradient_descent_maxit(normalize(X_tr), Y_tr, rate, max_iter, X_val, Y_val)
    print("If we don't use threshold and keep the learning rate = " , rate  )
    print("MSE of training data: ", Mean_Squared_Error(normalize(X_tr), Y_tr, weights) )
    print("MAE of training data: ", Mean_Absolute_Error(normalize(X_tr), Y_tr, weights) )
    print("MSE of validation data: ", Mean_Squared_Error(X_val, Y_val, weights) )
    print("MAE of validation data: ", Mean_Absolute_Error(X_val, Y_val, weights) )

    rate = 0.001
    threshold = -0.0001
    max_iter= 1000
    lambda_value = 5

    weights, t_lst, v_lst = ridge_gradient_descent(normalize(X_tr),  Y_tr, rate, lambda_value, threshold, X_val, Y_val)

    print("If we use threshold and keep the learning rate = " , rate  )
    print("MSE of training data: ", Mean_Squared_Error(normalize(X_tr),  Y_tr, weights) )
    print("MAE of training data: ", Mean_Absolute_Error(normalize(X_tr),  Y_tr, weights) )
    print("MSE of validation data: ", Mean_Squared_Error(X_val, Y_val, weights) )
    print("MAE of validation data: ", Mean_Absolute_Error(X_val, Y_val, weights) )

    weights, t_lst, v_lst = ridge_gradient_descent_maxit(X_tr, Y_tr, rate, lambda_value, max_iter, X_val, Y_val)
    print("If we don't use threshold and keep the learning rate = " , rate  )
    print("MSE of training data: ", Mean_Squared_Error(normalize(X_tr), Y_tr, weights) )
    print("MAE of training data: ", Mean_Absolute_Error(normalize(X_tr),  Y_tr, weights) )
    print("MSE of validation data: ", Mean_Squared_Error(X_val, Y_val, weights) )
    print("MAE of validation data: ", Mean_Absolute_Error(X_val, Y_val, weights) )


def visualisation_features():
    Xaxis = [10,100,1000,2048]
    Yaxis = [bestfeatures(10),bestfeatures(100),bestfeatures(1000),bestfeatures(2048)]
    plt.plot(Xaxis,Yaxis,color = "r")
    plt.show()

def normalize(x):
    mean = np.mean(x)
    stanDev = np.std(x)
    #print(mean, stanDev)
    for i in range(len(x)):
        x[i] = (x[i] - mean) / stanDev
    return x

def gradient_descent_last(X,y,learningrate, threshold):
    theta = theta_initial(X)
    N = len(X)
    rel_error = 100
    i = 0
    while rel_error > threshold and i < 100:
        gradients = 2/N * X.T.dot(X.dot(theta) - y)
        theta_new = theta - learningrate * gradients
        rel_error = abs(Mean_Squared_Error(X, y, theta) - Mean_Squared_Error(X, y, theta_new))
        print(rel_error, "\n")
        theta = theta_new
        i = i + 1
    return theta
    
def linear_regression_last(n):
    tp = str(n) + "_d_test.csv"
    trp = str(n) + "_d_train.csv"
    datatp = np.genfromtxt( tp, delimiter=",")
    X_org = datatp[:, 0:2]
    y_org = datatp[:, 2]
    datatrp = np.genfromtxt( trp, delimiter=",")
    X_org_val = datatrp[:, 0:2]
    y_org_val = datatrp[:, 2]

    y = np.reshape(y_org, (len(y_org), 1)) 
    X = np.c_[np.ones((len(X_org), 1)), X_org]

    y_val = np.reshape(y_org_val, (len(y_org_val), 1)) 
    X_val = np.c_[np.ones((len(X_org_val), 1)), X_org_val]
    ratelast = 0.001
    threshold = 0.001
    weights = gradient_descent_last(X, y, ratelast, threshold)
    E_in = Mean_Squared_Error(X, y, weights)
    E_out = Mean_Squared_Error(X_val, y_val, weights)
    return E_out - E_in

def run():
    x = [2, 5, 10, 100]
    E_difference = []
    for i in x:
        E_difference.append(linear_regression_last(i))
    print(E_difference)
    plt.plot(x, E_difference, color='red', linestyle='dashed', marker='o', markerfacecolor='blue')
    plt.xlabel('Dimension')
    plt.ylabel('E_out - E_in')
    plt.show()

nn = len(sys.argv)

if(nn ==0):
    linear_regression()
else:
    if(sys.argv[1] == "linear_regression"):
        linear_regression()
    if(sys.argv[1] == "ridge_regression"):
        ridge_regression()
    if(sys.argv[1] == "skl"):
        sklLibrary()
    if(sys.argv[1] == "features"):
        bestfeatures()
    if(sys.argv[1] == "visualisation_normal"):
        visualisation_normal()
    if(sys.argv[1] == "visualisation_features"):
        visualisation_features()
    if(sys.argv[1] == "last"):
        run()
