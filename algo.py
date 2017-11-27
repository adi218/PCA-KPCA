import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def pca(y, d):
    mean = np.zeros((y.shape[0], 1))
    for i in range(y.shape[1]):
        mean += y[:, i]/y.size
    centred_y = np.zeros(y.shape)
    for i in range(y.shape[1]):
        temp = y[:, i] - mean
        centred_y[:, i] = temp.T
    u, s, v = np.linalg.svd(centred_y)
    print(u.shape)
    plt.plot(s)
    plt.show()
    x = np.zeros((d, y.shape[1]))
    for i in range(x.shape[1]):
        x[:, i] = np.matmul(u[:, :d].T, (centred_y[:, i]))
    return mean, u, x


def kpca(k, d):
    i = np.eye(k.shape[1])
    o = np.ones(k.shape[1])/k.shape[1]
    mul = i - o
    k_n = np.matmul(np.matmul(mul, k), mul)
    print(k_n.shape)
    w, v = np.linalg.eig(k_n)
    # print(v)
    for i in range(v.shape[1]):
        v[:, i] = v[:, i]/w[i]
    temp = v[:, :d].T
    x = np.matmul(temp, k)
    return x


def kernel(y, sig):
    k = np.zeros((y.shape[1], y.shape[1]))
    for i in range(y.shape[1]):
        for j in range(y.shape[1]):
            temp = np.linalg.norm(y[:, i] - y[:, j], ord=2)**2/(2*(sig**2))
            k[i, j] = np.exp(-temp)
    return k


data = scipy.io.loadmat('dataset1.mat')
Y = np.asmatrix(data['Y'])
ker = kernel(Y, 0.6)
x = kpca(ker, 2)
plt.scatter(x[0, :], x[1, :], c='blue')
plt.show()
