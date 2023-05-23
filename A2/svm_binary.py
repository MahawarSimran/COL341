from typing import List
import numpy as np
import kernel
# import qpsolvers
import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cvxopt
from cvxopt import matrix
from cvxopt import solvers
from scipy.spatial.distance import cdist
from numpy.linalg import norm
import math
from tqdm import tqdm

def load(file):
    data = np.genfromtxt(file, delimiter=",")
    a,b = data.shape
    X_org = data[1:a, 1:b-1]
    y_org = data[1:a , b-1:b]
    for i in range(a-1):
        if(y_org[i][0] < 1):
            y_org[i][0] = -1.0
    # print(y_org.shape)
    # print(X_org.shape)
    return X_org,y_org

def wTx(alpha, norms, SupportVectors, x):
    m = len(SupportVectors)
    if m ==0:
        return 0
    temp = np.square(norms) + np.full((m, 1), norm(x)** 2) - 2*np.dot(SupportVectors, x)
    temp = np.exp(-0.05*temp)
    return np.sum(alpha*temp)

class Trainer:
    def __init__(self,kernel,C=None,**kwargs) -> None:
        self.kernel = kernel
        self.kwargs = kwargs
        self.C= C
        self.support_vectors:List[np.ndarray] = []
        self.accuracy = 0

    def linear_kernal(self,X, Y, c):
        P = matrix(np.dot(Y*X, (Y*X).T), tc='d')
        q = matrix(np.full((X.shape[0], 1), -1), tc='d')
        # -1*alphai <= 0 and alphai <= c
        G = matrix(
            np.vstack((-1*np.eye(X.shape[0]), np.eye(X.shape[0]))), tc='d')
        h = matrix(
            np.vstack((np.zeros((X.shape[0], 1)), c*np.ones((X.shape[0], 1)))), tc='d')
        A = matrix(Y.reshape(1, -1), tc='d')
        b = matrix(np.array([0]), tc='d')
        solvers.options['show_progress'] = False
        cvxopt_solver = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(cvxopt_solver['x'])
        Support = (alpha > 1e-4)
        Support = Support.flatten()
        w = np.dot((Y[Support] * alpha[Support]).T, X[Support])
        w = w.reshape(-1, 1)
        b = np.mean(Y[Support] - np.dot(X[Support], w))
        SupportVectors = [X[i] for i in range(len(alpha)) if alpha[i] > 1e-4]
        alpha = [x for x in alpha if x > 1e-4]
        return w, b, alpha, SupportVectors

    def linear_predict(self, X, Y, w, b):
        prediction = [1.0 if np.dot(x.reshape((1, -1)), w)[0][0] + b >= 0 else -1.0 for x in X]
        accuracy = 100*sum([(1 if Y[i][0] == prediction[i]
                        else 0) for i in range(len(Y))])/len(Y)
        print ("accuracy in linear for c =" , self.C , "= ")
        print (accuracy)
        self.accuracy = accuracy
        return np.array(prediction)
    def gaussss(self,x,z,gamma):
        L2norm= cdist(x,z,'euclidean')
        return np.exp(-(L2norm**2)*gamma)
    def Gaussfit(self,X,Y,g,c,a):
        G1 = np.diag(np.array([1]*Y.shape[0]))
        G2 = np.diag(np.array([-1]*Y.shape[0]))
        G = matrix(np.append(G1,G2,axis=0),tc='d')
        H = matrix(np.append(np.array([c]*Y.shape[0]),np.array([0]*Y.shape[0]),axis = 0),tc='d')
        Q = matrix(np.array([-1]*Y.shape[0]).T,tc='d')
        A = matrix(Y.T,tc='d')
        B = matrix(np.array(0).reshape((1,1)),tc='d')
        IP = []
        IP = self.kernel(X,gamma=g,alpha = a)
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
        b1 = np.dot(self.kernel(supportVectors[:,:-2],second = addx,gamma=g,alpha=a).T,(alpha_i*y_i)).min()
        b2 = np.dot(self.kernel(supportVectors[:,:-2],second=minusx,gamma=g,alpha=a).T,(alpha_i*y_i)).max()

        b = -(b1+b2)/2
        return b,supportVectors
    def GausianSVM_predict(self,X,Y,b,supportVectors,g,a):
        y_i = supportVectors[:,-2]
        alpha_i =supportVectors[:,-1]
        wx = np.dot(self.kernel(supportVectors[:,:-2],second= X, gamma=g,alpha=a).T, (alpha_i*y_i)) 
        prediction = wx+b

        count = 0
        for i in range(X.shape[0]):
            if (prediction[i] >0 and Y[i]==1) or (prediction[i] <=0 and Y[i]== -1):
                count+=1
        self.accuracy = (count*100)/X.shape[0]
        print("accuracy" ,(count*100)/X.shape[0])
        return np.array(prediction)
    
    def fit(self, train_data_path:str)->None:
        #TODO: implement
        #store the support vectors in self.support_vectors
        X,Y = load(train_data_path)
        alpha = 0.1
        gamma = 0.01
        C = 0.1
        for val,value in self.kwargs.items():
            if(val == "Q"):
                Q=value
            if(val == "gamma"):
                gamma  = value
            if(val == "alpha"):
                gamma  = value
        if(self.C!= None):
            C = self.C
        if self.kernel == kernel.linear:
            W,b,alp,sv = self.linear_kernal(X, Y, C)
            self.wm  =W
            self.b = b
            self.support_vectors = sv
        else:
            b,sv = self.Gaussfit(X, Y, gamma, C, alpha)
            self.b = b
            self.support_vectors = sv

    def predict(self, test_data_path:str)->np.ndarray:
        #TODO: implement
        #Return the predicted labels as a numpy array of dimension n_samples on test data
        gamma = 0.01
        alpha=1.0
        for val,value in self.kwargs.items():
            if(val == "gamma"):
                gamma = value
            if(val == "alpha"):
                alpha = value
            print("gamma = " , gamma)
        X,Y = load(test_data_path)
        if(self.b == None):
            print("Model not trained")
        if(self.kernel==None):
            self.kernel=kernel.linear
        if self.kernel == kernel.linear:
            return self.linear_predict(X, Y, self.wm, self.b)
        else:
            return self.GausianSVM_predict(X, Y, self.b, self.support_vectors, gamma,alpha)
        pass


def Accuracy():

    C = [0.01,0.1,1.0,10.0,100]
    Gamma = [0.001,0.01,0.1]

    BestAccuracy=[]
    GammaAtBest=[]
    for c in C:
        print("==========================================")
        print("--------------RBF---------------")
        A=[]
        mini=0
        g=0.1
        for gamma in Gamma:
            print("gamma: "+str(gamma))
            print("C: "+str(c))
            trainer1 = Trainer(kernel=kernel.rbf,C=c,gamma=gamma)
            trainer1.fit("bi_train.csv")
            trainer1.predict("bi_val.csv")
            if(trainer1.accuracy>mini):
                mini=trainer1.accuracy
                g=gamma
            A.append(trainer1.accuracy)
        BestAccuracy.append(mini)
        GammaAtBest.append(g)
        
        plt.plot(Gamma,A,color='green', linewidth = 3,marker='o', markerfacecolor='red', markersize=12)
        plt.xlabel("Gamma")
        plt.ylabel("Accuracy")
        plt.title("For C ="+str(c))
        plt.show()

    plt.plot(C,BestAccuracy,color='blue', linestyle='dashed', linewidth = 3,marker='o', markerfacecolor='red', markersize=12)
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.title("For Best Accuracy")
    plt.show()    

  
    print(" C                    Best gamma                Accuracy %   |")
    for i in range(len(C)):
        s="   {c}                      {gamma}                     {Accuracy}     |          "
        p=s.format(c=C[i],gamma=GammaAtBest[i],Accuracy=BestAccuracy[i])
        print(p)
    print("==============================================================")

def AccuracyLinear():

    C = [0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100]
    Gamma = [0.001,0.01,0.1]
    A=[]
    
    for c in C:
        
        print("C: "+str(c))
        trainer1 = Trainer(kernel=kernel.linear,C=c)
        trainer1.fit("bi_train.csv")
        trainer1.predict("bi_val.csv")
        
        A.append(trainer1.accuracy)  

    plt.plot(C,A,color='green', linestyle='dashed', linewidth = 3,marker='o', markerfacecolor='black', markersize=12)
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.title("For Best Accuracy")
    plt.show()    

    print(" C              Accuracy %   |")
    for i in range(len(C)):
        s="   {c}                {Accuracy}              "
        p=s.format(c=C[i],Accuracy=A[i])
        print(p)
    print("==============================================================")

# AccuracyLinear()
# trainer_lin = Trainer(kernel=kernel.linear,C= 10)
# trainer_lin.fit("bi_train.csv")
# trainer_lin.predict("bi_val.csv")

# trainer_rbf = Trainer(kernel=kernel.rbf,C= 1,gamma = 0.1)
# trainer_rbf.fit("bi_train.csv")
# trainer_rbf.predict("bi_val.csv")