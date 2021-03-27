# Beispiel 3b für NumMethDGl von Verena Alton

import numpy as np
from bokeh.plotting import figure, show


def ieuler(f, y0, t0, t1, h):
    """
    Improved Euler method for a differential equation: y' = f(t, y), y(t0)=y0.
    :param f: function
    :param y0: float or int, initial value y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval
    """
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    y_list = [0]*(N+1)
    y_list[0] = y0
    t_list = [0]*(N+1)
    t_list[0] = t0
    for k in range(N):
        xi = [0, 0]
        xi[0] = y_list[k]
        xi[1] = y_list[k] + 0.5*h*f(t, xi[1])
        y_list[k + 1] = y_list[k] + h * f(t+0.5*h, xi[1])
        t = t + h
        t_list[k+1] = t
    return y_list, t_list


def f(t, x):
    return 0.5*x*(10-x)


def y(t):
    return (10 * np.exp(5 * t)) / (np.exp(5 * t) + 9)


a, ta = ieuler(f, 1, 0, 1, 0.1)
b, tb = ieuler(f, 1, 0, 1, 0.05)
c, tc = ieuler(f, 1, 0, 1, 0.01)
print('Näherung bei t=1 mit h=0.1: ', a[-1])
print('Näherung bei t=1 mit h=0.05: ', b[-1])
print('Näherung bei t=1 mit h=0.01: ', c[-1])

ya = []
for t in ta:
    x = y(t)
    ya.append(x)

yb = []
for t in tb:
    x = y(t)
    yb.append(x)

yc = []
for t in tc:
    x = y(t)
    yc.append(x)

print('Exakte Lösung bei t=1: ', y(1))
print('Fehler bei t=1: ', a[-1]-y(1), b[-1]-y(1), c[-1]-y(1))

# Plot mit Bokeh
p = figure(title="Verbessertes Euler-Verfahren", x_axis_label='t', y_axis_label='y(t)')
p.line(ta, a, legend_label="h=0.1", line_width=2, line_color="green")
p.line(tb, b, legend_label="h=0.05", line_width=2, line_color="blue")
p.line(tc, c, legend_label="h=0.01", line_width=2, line_color="purple")
p.line(tc, yc, legend_label="exakt", line_width=2, line_color="red")
show(p)

