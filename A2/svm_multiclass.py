from typing import List
import numpy as np
from svm_binary import Trainer
import time
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cvxopt
import math
from cvxopt import matrix
from cvxopt import solvers
from scipy.spatial.distance import cdist
from tqdm import tqdm

thisdict= {}
alldict = {}

def gaussss(x,z,gamma):
    L2norm= cdist(x,z,'euclidean')
    return np.exp(-(L2norm**2)*gamma)

def load(file):
    data = np.genfromtxt(file, delimiter=",")
    a,b = data.shape
    X_org = data[1:a, 2:b]
    y_org = data[1:a , 1:2]
    for i in range(a-1):
        if(y_org[i][0] < 1):
            y_org[i][0] = -1.0
    # print(y_org.shape)
    # print(X_org.shape)
    return X_org,y_org

def loadclass(val1,val2, file):
    data = np.genfromtxt(file, delimiter=",")
    a,b = data.shape
    X_org = data[1:a, 2:b]
    y_org = data[1:a , 1:2]
    cnt = 0
    for i in range(a-1):
        if(y_org[i][0]  == val2 or y_org[i][0] == val1):
            cnt += 1
    X_train =np.zeros((cnt,b-2))
    Y_train = np.zeros((cnt,1))
    k =0
    for x in range(a-1):
        if(y_org[x][0]==val1) : 
            Y_train[k][0] += 1.0
            X_train[k]=np.array(X_org[x])
            k+=1
        elif(y_org[x][0]==val2):
            Y_train[k][0] -= 1.0 
            X_train[k]=np.array(X_org[x])
            k+=1
    return X_train,Y_train

def loadonebyall(val1, file):
    data = np.genfromtxt(file, delimiter=",")
    a,b = data.shape
    X_org = data[1:a, 2:b]
    y_org = data[1:a , 1:2]
    for x in range(a-1):
        if(y_org[x][0] == val1) : 
            y_org[x][0] = 1
        else:
            y_org[x][0] = -1
    return X_org, y_org
def Gaussfit(X,Y,gamma):
    G1 = np.diag(np.array([1]*Y.shape[0]))
    G2 = np.diag(np.array([-1]*Y.shape[0]))
    G = matrix(np.append(G1,G2,axis=0),tc='d')
    c = 1
    H = matrix(np.append(np.array([c]*Y.shape[0]),np.array([0]*Y.shape[0]),axis = 0),tc='d')
    Q = matrix(np.array([-1]*Y.shape[0]).T,tc='d')
    A = matrix(Y.T,tc='d')
    B = matrix(np.array(0).reshape((1,1)),tc='d')
    IP = []
    
    L2norm= cdist(X,X,'euclidean')
    IP = np.exp(-(L2norm**2)*gamma)
    P = matrix(IP*np.dot(Y,Y.T),tc='d')
    
    #-----------------------------------
    sol = solvers.qp(P,Q,G,H,A,B)
    alpha=np.array(sol['x'])
    #-----------------------------------
    
    threshold = 1e-5
    data = np.append(X,Y,axis = 1)
    data_with_alpha = np.append(data,alpha,axis = 1)

    supportVectors = data_with_alpha[np.where(data_with_alpha[:,-1] >= threshold)]
    # print("Total supports vectors")
    # print(supportVectors.shape)
    # print('\n')
    XY = np.append(X,Y,axis=1)
    addx = XY[np.where(XY[:,-1]==1)][:,:-1]
    minusx = XY[np.where(XY[:,-1]==-1)][:,:-1]
    #b
    y_i = supportVectors[:,-2]
    alpha_i = supportVectors[:,-1]
    b1 = np.dot(gaussss(supportVectors[:,:-2],addx,gamma).T,(alpha_i*y_i)).min()
    
    b2 = np.dot(gaussss(supportVectors[:,:-2],minusx,gamma).T,(alpha_i*y_i)).max()

    b = -(b1+b2)/2
    return b,supportVectors
class Trainer_OVO:
    sv = None
    b = None
    def __init__(self, kernel, C=None, n_classes = -1, **kwargs) -> None:
        self.kernel = kernel
        self.C = C
        self.n_classes = n_classes
        self.kwargs = kwargs
        self.svms = [] # List of Trainer objects [Trainer]
    
    def _init_trainers(self):
        #TODO: implement
        #Initiate the svm trainers 
        return  Trainer("rbf",C= 0.01,gamma = 0.1)  
    
    def fit(self, train_data_path:str, max_iter=None)->None:
        #TODO: implement
        #Store the trained svms in self.svm
        X,Y = load(train_data_path)
        gamma = 0.1
        c = 1.0
        for val, value in self.kwargs:
            if (val == "gamma"):
                gamma = value
        if(self.C != None):
            C = self.C
        if(self.kernel == "rbf"):
            for i in range(1,10):
                for j in range (i+1, 10):
                    Xread, Yread = loadclass(i, j,train_data_path)
                    b, sv = Gaussfit(Xread, Yread, gamma)
                    modell = [b,sv]
                    thisdict[(i,j)] = modell
        pass
    def predict(self, test_data_path:str)->np.ndarray:
        X,Y = load(test_data_path)
        m = X.shape[0]
        gamma= 0.1
        count_matrix = np.zeros((m,10))
        score_matrix = np.zeros((m,10))
        for i in tqdm(range(1,10)):
            for j in range(i+1,10):
                model = thisdict[(i,j)]
                b = model[0]
                supportVectors = model[1]
                y_i = supportVectors[:,-2]
                alpha_i =supportVectors[:,-1]
                wx = np.dot(gaussss(supportVectors[:,:-2], X, gamma).T, (alpha_i*y_i)) 
                prediction_score = wx+b
                for k in range(X.shape[0]):
                    if prediction_score[k] > 0:
                        score_matrix[k,i] += abs((prediction_score[k]))
                        count_matrix[k,i]+=1
                    else:
                        score_matrix[k,j] += abs((prediction_score[k]))
                        count_matrix[k,j]+=1
        final_pred = []
        correct_pred_count = 0
        for i in tqdm(range(X.shape[0])):
            cind = []
            max_value = count_matrix[i,:].max()
            for j in range(10):
                if max_value == count_matrix[i,j]:
                    cind.append(j)
            if len(cind)==1:
                final_pred.append(cind[0])
                if final_pred[i] == Y[i]:
                    correct_pred_count+=1   
            else:
                score_value = -math.inf
                for k in cind:
                    if score_matrix[i,k] > score_value:
                        score_value = score_matrix[i,k]
                        index = k
                final_pred.append(index)
                if final_pred[i] == Y[i][0]:
                    correct_pred_count+=1
        print("accuracy is ",(correct_pred_count*100)/Y.shape[0])
        #TODO: implement
        #Return the predicted labels
        pass

    
class Trainer_OVA:
    def __init__(self, kernel, C=None, n_classes = -1, **kwargs) -> None:
        self.kernel = kernel
        self.C = C
        self.n_classes = n_classes
        self.kwargs = kwargs
        self.svms = [] # List of Trainer objects [Trainer]
    
    def _init_trainers(self):
        #TODO: implement
        #Initiate the svm trainers 
        pass
    
    def fit(self, train_data_path:str, max_iter=None)->None:
        #TODO: implement
        #Store the trained svms in self.svms
        X,Y = load(train_data_path)
        gamma = 0.01
        c = 1.0
        for val, value in self.kwargs:
            if (val == "gamma"):
                gamma = value
        if(self.C != None):
            C = self.C
        if(self.kernel == "rbf"):
            for i in range(1,10):
                Xread, Yread = loadonebyall(i, train_data_path)
                b, sv = Gaussfit(Xread, Yread, gamma)
                modell = [b,sv]
                alldict[i]= modell
        pass
    def predict(self, test_data_path:str)->np.ndarray:
        X,Y = load(test_data_path)
        m = X.shape[0]
        gamma = 0.1
        score_matrix = np.zeros((m,10))
        for i in tqdm(range(1,10)):
                model = alldict[i]
                b = model[0]
                supportVectors = model[1]

                y_i = supportVectors[:,-2]
                alpha_i =supportVectors[:,-1]

                wx = np.dot(gaussss(supportVectors[:,:-2], X, gamma).T, (alpha_i*y_i)) 

                prediction_score = wx+b

                for k in range(X.shape[0]):
                    score_matrix[k,i] = ((prediction_score[k]))

        final_pred = []
        correct_pred_count = 0
        confusion_matrix = np.zeros((10,10))
        for i in tqdm(range(X.shape[0])):
            cind = []
            max_value = score_matrix[i,:].max()
            for j in range(1,10):
                if max_value == score_matrix[i,j]:
                    cind.append(j)
            if len(cind)==1:
                final_pred.append(cind[0])
                # print("final_pred[i] : ",final_pred[i])
                # print("Y: ",int(Y[i]))
                if final_pred[i] == Y[i]:
                    correct_pred_count+=1
                    confusion_matrix[int(Y[i][0]),int(Y[i][0])]+=1
                else:
                    confusion_matrix[int(final_pred[i]),int(Y[i])]+=1
                    
            else:
                # print(score_matrix[i,:])
                score_value = -math.inf
                index=0
                for k in cind:
                    if score_matrix[i,k] > score_value:
                        score_value = score_matrix[i,k]
                        index = k
                final_pred.append(index)
                if final_pred[i] == Y[i][0]:
                    correct_pred_count+=1
                    confusion_matrix[int(Y[i]),int(Y[i])]+=1
                else:
                    confusion_matrix[int(final_pred[i]),int(Y[i])]+=1
                    
        print("when c = 1.0 accuracy is ",(correct_pred_count*100)/Y.shape[0])
        return confusion_matrix
trainer_one = Trainer_OVO("rbf")
trainer_one.fit("multi_train.csv")
print("OVO")
trainer_one.predict("multi_val.csv")
trainer_two = Trainer_OVA("rbf")
print("OVA")
trainer_two.fit("multi_train.csv")
trainer_two.predict("multi_val.csv")
# trainer_one.predict("multi_val.csv")


