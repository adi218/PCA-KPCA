import numpy as np
import scipy.io
import random
import matplotlib.pyplot as plt
from sklearn import svm


def pca(y):
    mean = np.zeros((y.shape[0], 1))
    for i in range(y.shape[1]):
        mean += y[:, i]/y.size
    centred_y = np.zeros(y.shape)
    for i in range(y.shape[1]):
        temp = y[:, i] - mean
        centred_y[:, i] = temp.T
    u, s, v = np.linalg.svd(centred_y)
    print(s)
    plt.plot(s)
    plt.show()
    x = np.zeros((u.shape[0], y.shape[1]))
    # for i in range(x.shape[1]):
    #     x[:, i] =
    return mean, u, x

def kpca(k):
    i = np.eye(k.size)
    print(np.ones(k.shape[1]))
    mul = (i - (np.dot(np.ones(k.shape[1]), np.ones(k.shape[1]).T)/k.size))
    k_n = np.dot(np.dot(mul, k), mul)
    w, v = np.linalg.eig(k_n)
    n = np.linalg.norm(w, ord=2)
    print(n)

def kernel(y, sig):
    k = np.zeros((y.shape[1], y.shape[1]))
    for i in range(y.shape[1]):
        for j in range(y.shape[1]):
            temp = -np.linalg.norm(y[:, i] - y[:, j], 2)/2*pow(sig, 2)
            k[i, j] = np.exp(temp)
    return k




data = scipy.io.loadmat('dataset1.mat')
Y = np.asmatrix(data['Y'])
ker = kernel(Y, 2)
print(ker[1, 1])
kpca(ker)
