# Beispiel 2b für NumMethDGl von Verena Alton

import numpy as np
from bokeh.plotting import figure, show


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
    t_list = [0] * (N + 1)
    t_list[0] = t0
    y = np.zeros((len(f), N + 1))
    y[:, 0] = y0
    for k in range(N):
        for i in range(len(f)):
            y[i, k + 1] = y[i, k] + h * f[i](t, y[:, k])
            # Nur fuer Raeuber-Beute-Modell: Werte duerfen nicht negativ werden
            if y[i][k + 1] < 0:
                y[i][k + 1] = 0
            t = t + h
            t_list[k + 1] = t
    return y, t_list


def raeuber(t, x):
    alpha = 1,
    beta = 0.1,
    return -(1 - 0.1*x[1])*x[0]


def beute(t, x):
    gamma = 4,
    delta = 1,
    return (4-1*x[0])*x[1]


y, t = euler_system([raeuber, beute], [3, 5], 0, 10, 0.001)

print('Räuber: ', y[0])
print('Beute: ', y[1])

# Plot mit Bokeh
p = figure(title="Räuber-Beute-Modell", x_axis_label='t')
p.line(y[1], y[0], legend_label="Räuber", line_width=2, line_color="red")
#p.line(t, y[1], legend_label="Beute", line_width=2, line_color="green")
show(p)


"""
Old version - System of exactly 2:

def euler_2system(fy, fz, y0, z0, t0, t1, h):
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    y = [0]*(N+1)
    z = [0]*(N+1)
    y[0] = y0
    z[0] = z0
    for k in range(N):
        y[k + 1] = y[k] + h * fy(t, [y[k], z[k]])
        z[k + 1] = z[k] + h * fz(t, [y[k], z[k]])
        # Nur fuer Raeuber-Beute-Modell: Werte duerfen nicht negativ werden
        if y[k+1] < 0:
            y[k+1] = 0
        if z[k+1] < 0:
            z[k+1] = 0
        t = t + h
    return y, z


y, z = euler_2system(raeuber, beute, 3, 5, 0, 10, 1)
"""
