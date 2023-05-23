import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib import style
from run import *

if(len(sys.argv)<5):
    linear_regression()
else:
    train = sys.argv[1]
    validate = sys.argv[2]
    test = sys.argv[3]
    out = sys.argv[4]
    typeofregression = sys.argv[5]

def linearreg(a,b,c,d):
    data = np.genfromtxt(a, delimiter=",")
    X_org = data[:, 2:]
    y_org = data[:, 1]
    Y_tr = np.reshape(y_org, (len(y_org), 1)) 
    X_tr = np.c_[np.ones((len(X_org), 1)), X_org]
    data = np.genfromtxt(b, delimiter=",")
    X_org_val = data[:, 2:]
    y_org_val = data[:, 1]
    Y_val = np.reshape(y_org_val, (len(y_org_val), 1)) 
    X_val = np.c_[np.ones((len(X_org_val), 1)), X_org_val]
    data = np.genfromtxt(c, delimiter=",")
    X_org_test = data[:, 1:]
    samplename = data[:, 0]
    X_test = np.c_[np.ones((len(X_org_val), 1)), X_org_val]
    rate = 0.001
    max_iter= 1000
    threshold = -0.0001

    weightstest, t_test, v_test = gradient_descent(X_tr, Y_tr, rate, threshold, X_val, Y_val)
    prediction = X_test.dot(weightstest)
    f = open(d, "w")
    print(prediction)
    for i in range(len(prediction)):
        st = str(prediction[i]) + "\n"
        f.write(st)
    f.close()


def ridgereg(a,b,c,d):
    data = np.genfromtxt(a, delimiter=",")
    X_org = data[:, 2:]
    y_org = data[:, 1]
    Y_tr = np.reshape(y_org, (len(y_org), 1)) 
    X_tr = np.c_[np.ones((len(X_org), 1)), X_org]
    data = np.genfromtxt(b, delimiter=",")
    X_org_val = data[:, 2:]
    y_org_val = data[:, 1]
    Y_val = np.reshape(y_org_val, (len(y_org_val), 1)) 
    X_val = np.c_[np.ones((len(X_org_val), 1)), X_org_val]
    data = np.genfromtxt(c, delimiter=",")
    X_org_test = data[:, 1:]
    samplename = data[:, 0]
    X_test = np.c_[np.ones((len(X_org_val), 1)), X_org_val]
    
    rate = 0.001
    threshold = -0.0001
    max_iter= 1000
    lambda_value = 5

    weightstest, t_test, v_test = ridge_gradient_descent(X_tr, Y_tr, rate,lambda_value, threshold, X_val, Y_val)
    prediction = X_test.dot(weightstest)
    f = open(d, "w")
    print(prediction)
    for i in range(len(prediction)):
        st = str(prediction[i]) + "\n"
        f.write(st)
    f.close()

if(typeofregression == "linear"):
    linearreg(train, validate, test, out)
else:
    linearreg(train, validate, test, out)

