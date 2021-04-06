import numpy as np
from bokeh.plotting import figure, show

t0 = 0
t1 = 10
y0 = 1
h = 0.1


def f(t, y):
    dy = [0] * len(y)
    dy[0] = -2 * t * y[0] ** 2
    return dy


def forward_euler(f, y0, t0, t1, h):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    y = np.zeros((len(y0), N + 1))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    y[:, 0] = y0
    for k in range(N):
        for i in range(len(y0)):
            y[i, k + 1] = y[i, k] + h * f(t, y[:, k])[i]
            t = t + h
            t_list[k + 1] = t
    return y, t_list


def forward_euler2(f, y0, t0, t1, h):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    n_ode = len(y0)
    nh = int((t1 - t0) / h)
    t = t0
    y = y0
    t_list = [t]
    y_list = [y[0]]
    for i in range(nh):
        for j in range(n_ode):
            y[j] = y[j] + h * f(t, y)[j]
        t = t + h
        t_list.append(t)
        for j in range(len(y)):
            y_list.append(y[j])
    return y_list, t_list


def backward_euler(f, y0, t0, t1, h):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    n_ode = len(y0)
    nh = int((t1 - t0) / h)
    t = t0
    y = y0
    t_list = [t]
    y_list = [y[0]]
    for i in range(nh):
        for j in range(n_ode):
            y_derived = f(t + h, y)[j] / (1 + h)
            y[j] = y[j] + h * y_derived
        t = t + h
        t_list.append(t)
        for j in range(len(y)):
            y_list.append(y[j])
    return y_list, t_list


def crank_nicolson(f, y0, t0, t1, h):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    n_ode = len(y0)
    nh = int((t1 - t0) / h)
    t = t0
    y = y0
    t_list = [t]
    y_list = [y[0]]
    for i in range(nh):
        for j in range(n_ode):
            y_derived = f(t + h, y)[j] / (1 + h)
            y[j] = y[j] + h / 2 * (y_derived + f(t, y)[j])
        t = t + h
        t_list.append(t)
        for j in range(len(y)):
            y_list.append(y[j])
    return y_list, t_list


def add(vector, scalar):
    newvector = [0] * len(vector)
    for i in range(len(vector)):
        newvector[i] = vector[i] + scalar
    return newvector


def runge_kutta_3(f, y0, t0, t1, h):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    n_ode = len(y0)
    nh = int((t1 - t0) / h)
    t = t0
    y = y0
    t_list = [t]
    y_list = [y[0]]
    for i in range(nh):
        for j in range(n_ode):
            k1 = f(t, y)[j]
            k2 = f(t + h / 2, add(y, h * k1 / 2))[j]
            k3 = f(t + h, add(y, h * (-k1 + 2 * k2)))[j]
            y[j] = y[j] + h / 6 * (k1 + 4 * k2 + k3)
        t = t + h
        t_list.append(t)
        for j in range(len(y)):
            y_list.append(y[j])
    return y_list, t_list


def runge_kutta(f, y0, t0, t1, h):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    y = np.zeros((len(y0), N + 1))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    y[:, 0] = y0
    for k in range(N):
        for i in range(len(y0)):
            k1 = f(t, y[:, k])[i]
            k2 = f(t + h / 2, add(y[:, k], h * k1 / 2))[i]
            k3 = f(t + h, add(y[:, k], h * (-k1 + 2 * k2)))[i]
            y[i, k + 1] = y[i, k] + h / 6 * (k1 + 4 * k2 + k3)
            t = t + h
            t_list[k + 1] = t
    return y, t_list


def adams_bashforth_3(f, y0, t0, t1, h):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    N = int(np.ceil((t1 - t0) / h))
    y = np.zeros((len(y0), N + 1))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    t_list[1] = t0 + h
    t_list[2] = t0 + 2*h
    t = t0 + 2*h
    y[:, 0] = y0
    y_start, s = forward_euler(f, y0, t0, t1, h)
    y[:, 1] = y_start[:, 1]
    y[:, 2] = y_start[:, 2]
    for k in range(2, N):
        for i in range(len(y0)):
            y[i, k + 1] = y[i, k] + h * ((23 / 12 * f(t, y[:, k])[i]) - (16 / 12 * f(t - h, y[:, k - 1])[i]) + (
                        5 / 12 * f(t - 2 * h, y[:, k - 2])[i]))
            t = t + h
            t_list[k + 1] = t
    return y, t_list


y1, s1 = forward_euler(f, [y0], t0, t1, h)
y2, s2 = backward_euler(f, [y0], t0, t1, h)
y3, s3 = crank_nicolson(f, [y0], t0, t1, h)
y4, s4 = runge_kutta(f, [y0], t0, t1, h)
y5, s5 = adams_bashforth_3(f, [y0], t0, t1, h)


p = figure(title="Verschiedene numerische Verfahren", x_axis_label='t', y_axis_label='y(t)')
p.line(s1, y1[0], legend_label="Forward Euler", line_width=2, line_color="green")
p.line(s2, y2, legend_label="Backward Euler", line_width=2, line_color="blue")
p.line(s3, y3, legend_label="Cranc-Nicolson", line_width=2, line_color="purple")
p.line(s4, y4[0], legend_label="Runge-Kutta Ord. 3", line_width=2, line_color="red")
p.line(s5, y5[0], legend_label="Adam-Bashforth Ord. 3", line_width=2, line_color="cyan")
show(p)
