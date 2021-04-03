# Beispiel 2a für NumMethDGl von Verena Alton

import numpy as np
from bokeh.plotting import figure, show

h = 0.1


def euler(f, y0, t0, t1, h):
    """
    Explicit Euler method for a differential equation: y' = f(t, y).
    :param f: function
    :param y0: float or int, initial value y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval
    """
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    y = [0]*(N+1)
    t_list = [0]*(N+1)
    t_list[0] = t0
    y[0] = y0
    for k in range(N):
        y[k + 1] = y[k] + h * f(t, y[k])
        t = t + h
        t_list[k+1] = t
    return y, t_list


def euler_system(f, y0, t0, t1, h):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval
    """
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    y = np.zeros((len(y0), N + 1))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    y[:, 0] = y0
    for k in range(N):
        for i in range(len(y0)):
            y[i, k + 1] = y[i, k] + h * f[i](t, y[:, k])
            t = t + h
            t_list[k + 1] = t
    return y, t_list


def f(t, y):
    return y


y, t_list = euler_system([f], [1], 1, 4, h)
print('Annäherung: ', y)

exact = []
for t in t_list:
    exact.append(np.exp(t - 1))

print('Exakte Werte: ', exact)


q = figure(title="Explizites Euler-Verfahren", x_axis_label='t')
q.line(t_list, y[0], legend_label="Euler-Annäherung y'=y, y(1)=1", line_width=2, line_color="purple")
q.line(t_list, exact, legend_label="e^(t-1)", line_width=2, line_color="blue")
show(q)


