
import numpy as np
import matplotlib.pyplot as plt

def phi(d):
    x=2.0000
    for i in range(10):
        x = pow(1+x,1/(d+1))
    return x

def golden_samples(seed, d, n):

    # seed = 0.5
    # d = 2
    # n=16

    g = phi(2)
    alpha = np.zeros(d)
    for j in range(d):
        alpha[j] = pow(1/g,j+1) %1
    z = np.zeros((n, d))

    for i in range(n):
        z[i] = (seed + alpha*(i+1)) %1
    #     plt.scatter(z[i][0], z[i][1])
    # plt.show()

    return z

def golden_rejection_disk(seed, d, n):
    g = phi(2)
    alpha = np.zeros(d)
    for j in range(d):
        alpha[j] = pow(1/g,j+1) %1
    z = np.zeros((n * 3, d))
    res = np.zeros((n, d))
    count = 0
    for i in range(n * 3):
        z[i] = (seed + alpha*(i+1)) %1
        z[i] -= 0.5
        if (z[i][0]**2 + z[i][1]**2) < 0.5**2:
            res[count] = z[i] * 2
            count += 1
            # plt.scatter(z[i][0], z[i][1])
            if count == n:
                break
    # plt.show()
    return res