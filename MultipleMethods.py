import numpy as np
import sympy as sp
from bokeh.plotting import figure, show


def newton(f, x0, epsilon, max_iter):
    """Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    """
    xn = x0
    Df = deriv(f)
    for n in range(0, max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after', n, 'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn / Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None


def f(x):
    return x ** 2


def g(t, y):
    return y


def deriv(f):
    h = 1e-5
    return lambda x: (f(x + h) - f(x - h)) / (2 * h)


# xn = newton(f,1,1e-8,20)
# print(xn)


def forward_euler(f, y0, t0, t1, h):
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


def backward_euler(f, y0, t0, t1, h):
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
            y[i, k + 1] = y[i, k] + h * f[i](t+h, y[:, k + 1])
            t = t + h
            t_list[k + 1] = t
    return y, t_list


def feval(funcName, *args):
    return eval(funcName)(*args)


def mult(vector, scalar):
    newvector = [0] * len(vector)
    for i in range(len(vector)):
        newvector[i] = vector[i] * scalar
    return newvector


def backwardEuler(f, y0, t0, t1, h):
    n_ode = len(y0)
    N = int((t1 - t0) / h)

    t = t0
    y = y0

    t_list = [t]
    y_list = [y[0]]

    for i in range(N):
        yprime = feval(f, t + h, y)

        yp = mult(yprime, (1 / (1 + h)))

        for j in range(n_ode):
            y[j] = y[j] + h * yp[j]

        t = t + h
        t_list.append(t)

        for j in range(len(y)):
            y_list.append(y[j])  # Saves all new y's

    return y_list, t_list


def myFunc(x, y):
    '''
    We define our ODEs in this function.
    '''
    dy = [0] * len(y)
    dy[0] = 3 * (1 + x) - y[0]
    return dy


h = 0.2
x = [1.0, 2.0]
yinit = [4.0]

[ts, ys] = backwardEuler('myFunc', yinit, 1, 2, h)
print(ts, ys)

x, s = forward_euler([g], [1], 1, 4, 0.1)
y, t = backwardEuler('g', [1], 1, 4, 0.1)
print(x)
print(y)
