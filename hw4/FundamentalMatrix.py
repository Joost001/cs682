import numpy as np


def compute_fundamental(x, y):
    n = x.shape[1]

    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x[0, i] * y[0, i], x[0, i] * y[1, i], x[0, i] * y[2, i],
                x[1, i] * y[0, i], x[1, i] * y[1, i], x[1, i] * y[2, i],
                x[2, i] * y[0, i], x[2, i] * y[1, i], x[2, i] * y[2, i]]

    U, S, V=np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    return F


def fundamental_matrix(x, y):
    un_normalized_x = x
    un_normalized_y = y
    n = x.shape[1]

    mean_1=np.mean(x[:,:1])
    std_1 =np.std(x[:,:1])
    mean_2=np.mean(x[:,1:])
    std_2 =np.std(x[:,1:])
    x[:, :1] = (x[:, :1] - mean_1) / std_1
    x[:, 1:] = (x[:, 1:] - mean_2) / std_2

    mean_1=np.mean(y[:,:1])
    std_1 =np.std(y[:,:1])
    mean_2=np.mean(y[:,1:])
    std_2 =np.std(y[:,1:])
    y[:, :1] = (y[:, :1] - mean_1) / std_1
    y[:, 1:] = (y[:, 1:] - mean_2) / std_2

    # compute F with the normalized coordinates
    F=compute_fundamental(x,y)
    F_unnormalized = compute_fundamental(un_normalized_x, un_normalized_y)

    return F, F_unnormalized