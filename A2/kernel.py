import numpy as np

def linear(X: np.ndarray, **kwargs)-> np.ndarray:
    Y = None
    for key, value in kwargs.items():
        if(key=="second"):
            Y = value
    if Y is None:
        Y = X
    assert X.ndim == 2
    assert Y.ndim == 2
    kernel_matrix = X @ Y.T
    return kernel_matrix

def polynomial(X:np.ndarray,**kwargs)-> np.ndarray:
    Y = None
    for key, value in kwargs.items():
        if(key=="second"):
            Y = value
    if Y is None:
        Y = X
    assert X.ndim == 2
    assert Y.ndim == 2
    d = 2
    c = 1
    a = 1
    kernel_matrix = a*(X @ Y.T) + c
    kernel_matrix = kernel_matrix**2
    return kernel_matrix

def rbf(X:np.ndarray,**kwargs)-> np.ndarray:
    gamma = 0.01
    Y = None
    for key, value in kwargs.items():
        if (key=="gamma"):
            gamma = float(value)
        elif(key=="second"):
            Y = value
    if Y is None:
        Y = X
    assert X.ndim == 2
    assert Y.ndim == 2
    X_2 = (X**2).sum(axis=1).reshape(-1,1)@np.ones((1,Y.shape[0]))
    Y_2 = (Y**2).sum(axis=1).reshape(-1,1)@np.ones((1,X.shape[0]))
    kernel_matrix = np.exp(-gamma*(X_2 + Y_2.T - 2*X@Y.T))
    return kernel_matrix


def sigmoid(X:np.ndarray,**kwargs)-> np.ndarray:
    Y = None
    for key, value in kwargs.items():
        if(key=="second"):
            Y = value
    if Y is None:
        Y = X
    assert X.ndim == 2
    assert Y.ndim == 2
    c = 1
    a = 1
    kernel_matrix = a*(X @ Y.T) + c
    kernel_matrix = np.tanh(kernel_matrix)
    return kernel_matrix

def laplacian(X:np.ndarray,**kwargs)-> np.ndarray:
    gamma = 0.01
    Y = None
    for key, value in kwargs.items():
        if (key=="gamma"):
            gamma = float(value)
        elif(key=="second"):
            Y = value
    if Y is None:
        Y = X
    assert X.ndim == 2
    assert Y.ndim == 2
    X_2 = X.sum(axis=1).reshape(-1,1)@np.ones((1,Y.shape[0]))
    Y_2 = Y.sum(axis=1).reshape(-1,1)@np.ones((1,X.shape[0]))
    kernel_matrix = np.exp(-gamma*(np.linalg.norm(X_2 - Y_2.T)))
    return kernel_matrix


# import numpy as np
# import math
# from scipy.spatial.distance import cdist
# # Do not change function signatures
# #
# # input:
# #   X is the input matrix of size n_samples x n_features.
# #   pass the parameters of the kernel function via kwargs.
# # output:
# #   Kernel matrix of size n_samples x n_samples 
# #   K[i][j] = f(X[i], X[j]) for kernel function f()

# def linear(X: np.ndarray, **kwargs)-> np.ndarray:
#     assert X.ndim == 2
#     kernel_matrix = X @ X.T
#     return kernel_matrix

# def polynomial(X:np.ndarray,**kwargs)-> np.ndarray:
#     d = 2
#     c = 1
#     alp = 1
#     assert X.ndim == 2
#     linear_matrix = alp*(X @ X.T) + c
#     kernel_matrix = linear_matrix**2
#     return kernel_matrix

# def rbf(X:np.ndarray,**kwargs)-> np.ndarray:
#     assert X.ndim == 2
#     gamma = 0.01
#     for val, value in kwargs.items():
#         if(val == gamma):
#             gamma = float(value)
#     L2norm = cdist(X,X,'euclidean')
#     return np.exp(-(L2norm**2)*gamma)


# def sigmoid(X:np.ndarray,**kwargs)-> np.ndarray:
#     assert X.ndim == 2
#     gamma = 1.0
#     alp = 0.01
#     for val, value in kwargs.items():
#         if(val == gamma):
#             gamma = float(value)
#         if(val == alp):
#             gamma = float(value)
#     m=X.shape[0]
#     linear_matrix = X @ X.T
#     kernel_matrix = math.tanh(alp + gamma * linear_matrix)
#     return kernel_matrix

# def laplacian(X:np.ndarray,**kwargs)-> np.ndarray:
#     assert X.ndim == 2
#     gamma = 0.01
#     for val, value in kwargs.items():
#         if(val == gamma):
#             gamma = float(value)
#     m=X.shape[0]
#     kernel_matrix =np.zeroes(m,m)
#     i=0
#     X_2 = X.sum(axis=1).reshape(-1,1)@np.ones((1,X.shape[0]))
#     kernel_matrix = np.exp(-gamma*(np.linalg.norm(X_2 - X_2.T)))
#     return kernel_matrix