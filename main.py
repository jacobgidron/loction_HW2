import itertools

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/


def trilateration_model(controls, location):
    return np.linalg.norm((controls.T - location).T, axis=0)  # matrix is 2*m


def trilateration_residual(controls, observations, location):
    m_l = trilateration_model(controls, location)
    return m_l - observations


# def gen_controls(size,n=1000):
#     m = (np.array([i for i in itertools.product([1,-1],[1,-1])])*n).T
#     l = np.array([9500,500])
if __name__ == '__main__':
    n = 10_000
    m = (np.array([i for i in itertools.product([1, -1], [1, -1])]) * n).T
    l = np.array([9500, 500])
    m_l = trilateration_model(controls=m, location=l)
    xlist = np.arange(-20_000, 20_000 + 100, 100)
    ylist = np.arange(-20_000, 20_000 + 100, 100)
    norm_sq = lambda x: np.inner(x, x)
    error = lambda x, y: norm_sq(trilateration_residual(location=np.array([x, y]), controls=m, observations=m_l))
    error_v = np.vectorize(error)
    X, Y = np.meshgrid(xlist, ylist)
    # Z = np.sqrt(X ** 2 + Y ** 2)
    Z = error_v(X, Y)
    # plt.contourf(X, Y, Z)
    dots = np.zeros_like(Z)
    # plt.style.use('ggplot')
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.contour(X, Y, Z,levels = 120,zorder=-1)
    plt.scatter(*(m.tolist()), s=200,c='orange',zorder=1)
    plt.scatter(*l, s=200, c='r',zorder=-1)
    r = minimize(lambda x: error(x[0], x[1]), x0=np.array([-12000, -12000]), method='Nelder-Mead', tol=1e-6,
                 options={'return_all': True, 'disp': True})
    plt.plot(*zip(*r.allvecs), c='#3d3b3b', marker="o", markersize=6, zorder=1, markerfacecolor='#801224')
    # plt.plot(zip(*r.allvecs), s=200, c='r',zorder=1)
    plt.savefig(rf"D:\oneDrive\OneDrive - mail.tau.ac.il\Desktop\universty\semester 7\gps_algo\HW_3\g2_{np.random.randint(100)}")
    plt.show()
