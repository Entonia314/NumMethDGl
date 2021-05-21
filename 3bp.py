# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:13:29 2021

@author: Leo
"""
import numpy as np

G = 10
mass=[1,1,1]

def f(t, y):
    g = G
    m = mass
    d0 = ((-g * m[0] * m[1] * (y[0] - y[1]) / np.linalg.norm(y[0] - y[1]) ** 3) +
          (-g * m[0] * m[2] * (y[0] - y[2]) / np.linalg.norm(y[0] - y[2]) ** 3)) / m[0]
    d1 = ((-g * m[1] * m[2] * (y[1] - y[2]) / np.linalg.norm(y[1] - y[2]) ** 3) + (
            -g * m[1] * m[0] * (y[1] - y[0]) / np.linalg.norm(y[1] - y[0]) ** 3)) / m[1]
    d2 = ((-g * m[2] * m[0] * (y[2] - y[0]) / np.linalg.norm(y[2] - y[0]) ** 3) + (
            -g * m[2] * m[1] * (y[2] - y[1]) / np.linalg.norm(y[2] - y[1]) ** 3)) / m[2]
    return np.array([d0, d1, d2])


def fpi(f,t,yk,h,steps):
    x=yk
    while steps>0:
        x=(1+h)*x + h**2 * f(t,x)
        steps += -1

def forward_euler(f, y0, t0, t1, h):
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    v = np.zeros((len(y0), N + 1, 2))
    y = np.zeros((len(y0), N + 1, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    for k in range(N):
        for i in range(len(y0)):         
            v[i, k + 1, :] = v[i, k, :] + h * f(t, y[:, k + 1, :])[i]
            y[i, k + 1, :] = y[i, k, :] + h * v[i, k + 1, :]
            t = t + h
            t_list[k + 1] = t
    return y, v, t
    
def backward_euler(f, y0, t0, t1, h):
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    y = np.zeros((len(y0), N + 1, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    y[:, 0, :] = y0[:, 0, :]
    #y[:, k+1, :] = (1+h)*y[:, k, :] + h**2 f(t,y[:, k+1, :])
    for k in range(N):
        y[:, k+1, :] = fpi(f,t,y[:, k, :])
        t = t+h
        t_list[k + 1] = t
    return y, t_list
        
    