import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

mass = [1, 1, 1]
g = 1
p51 = 0.347111
p52 = 0.532728
inits3 = np.array([[[-1, 0], [p51, p52]],
                   [[1, 0], [p51, p52]],
                   [[0, 0], [-2 * p51, -2 * p52]]])



def f(y):
    d0 = ((-g * mass[0] * mass[1] * (y[0] - y[1]) / np.linalg.norm(y[0] - y[1]) ** 3) +
          (-g * mass[0] * mass[2] * (y[0] - y[2]) / np.linalg.norm(y[0] - y[2]) ** 3)) / mass[0]
    d1 = ((-g * mass[1] * mass[2] * (y[1] - y[2]) / np.linalg.norm(y[1] - y[2]) ** 3) + (
            -g * mass[1] * mass[0] * (y[1] - y[0]) / np.linalg.norm(y[1] - y[0]) ** 3)) / mass[1]
    d2 = ((-g * mass[2] * mass[0] * (y[2] - y[0]) / np.linalg.norm(y[2] - y[0]) ** 3) + (
            -g * mass[2] * mass[1] * (y[2] - y[1]) / np.linalg.norm(y[2] - y[1]) ** 3)) / mass[2]
    return np.array([d0, d1, d2])


def flong(y):
    d00 = ((-g * mass[0] * mass[1] * (y[0] - y[2]) / np.linalg.norm([y[0] - y[2], y[1] - y[3]]) ** 3) +
          (-g * mass[0] * mass[2] * (y[0] - y[4]) / np.linalg.norm([y[0] - y[4], y[1] - y[5]]) ** 3)) / mass[0]
    d01 = ((-g * mass[0] * mass[1] * (y[1] - y[3]) / np.linalg.norm([y[0] - y[2], y[1] - y[3]]) ** 3) + (
            -g * mass[0] * mass[2] * (y[1] - y[5]) / np.linalg.norm([y[0] - y[4], y[1] - y[5]]) ** 3)) / mass[0]
    d10 = ((-g * mass[1] * mass[2] * (y[2] - y[4]) / np.linalg.norm([y[2] - y[4], y[3] - y[5]]) ** 3) + (
            -g * mass[1] * mass[0] * (y[2] - y[0]) / np.linalg.norm([y[0] - y[2], y[1] - y[3]]) ** 3)) / mass[1]
    d11 = ((-g * mass[1] * mass[2] * (y[3] - y[5]) / np.linalg.norm([y[2] - y[4], y[3] - y[5]]) ** 3) + (
            -g * mass[1] * mass[0] * (y[3] - y[1]) / np.linalg.norm([y[0] - y[2], y[1] - y[3]]) ** 3)) / mass[1]
    d20 = ((-g * mass[2] * mass[0] * (y[4] - y[0]) / np.linalg.norm([y[0] - y[4], y[1] - y[5]]) ** 3) + (
            -g * mass[2] * mass[1] * (y[4] - y[2]) / np.linalg.norm([y[2] - y[4], y[3] - y[5]]) ** 3)) / mass[2]
    d21 = ((-g * mass[2] * mass[0] * (y[5] - y[1]) / np.linalg.norm([y[0] - y[4], y[1] - y[5]]) ** 3) + (
            -g * mass[2] * mass[1] * (y[5] - y[3]) / np.linalg.norm([y[2] - y[4], y[3] - y[5]]) ** 3)) / mass[2]
    return np.array([d00, d01, d10, d11, d20, d21])


def backward_euler_equation(y, yn, h, f):
    ret = y - yn - h*f(y)
    return ret


def backward_euler_scipy(f, y0, t0, t1, h):
    N = int(np.floor((t1 - t0)/h))
    t = t0
    v = np.zeros((N + 1, 6))
    y = np.zeros((N + 1, 6))
    y_start = y0[:, 0, :]
    v_start = y0[:, 1, :]
    y0_reshape = y_start.reshape(6,)
    v0_reshape = v_start.reshape(6,)
    t_list = [0] * (N + 1)
    t_list[0] = t0
    v[0, :] = v0_reshape
    y[0, :] = y0_reshape

    for k in range(N):
        v[k + 1, :] = fsolve(backward_euler_equation, y[k, :], (y[k, :], h, flong))
        y[k + 1, :] = y[k, :] + h * v[k + 1, :]
        t = t + h
        t_list[k + 1] = t
    print(y[1, :])
    y_ret = np.zeros((len(y0), N + 1, 2))
    y_ret[:, :, 0] = y[:, ::2].transpose()
    y_ret[:, :, 1] = y[:, 1::2].transpose()
    print(y_ret[:, 1, :])
    return y_ret, t_list


#print(f(inits3[:, 0, :]))
#init_flat = np.array([-1, 0, 1, 0, 0, 0])
#a = fsolve(backward_euler_equation, init_flat, (init_flat, 0.1, flong))
b = backward_euler_scipy(f, inits3, 0, 10, 0.1)



