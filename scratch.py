import numpy as np

p71 = 0.080584
p72 = 0.588836
inits7 = np.array([[[-1, 0], [p71, p72]],
                   [[1, 0], [p71, p72]],
                   [[0, 0], [-2 * p71, -2 * p72]]])

mass = [1, 1, 0.0000000000001]
mass_szenario4 = [0, 0, 0.00000001]
G = 1
t_start = 0
t_stop = 10


[x0, y0], [v0, w0] = [1, 2], [3, 4]
[x1, y1], [v1, w1] = [2, 2], [0.5, -0.5]
[x2, y2], [v2, w2] = [-3, -3], [1, -1]
init = np.array([[[x0, y0], [v0, w0]], [[x1, y1], [v1, w1]], [[x2, y2], [v2, w2]]])



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


def newton_raphson(f, g, x0, e, N):
    """
    Numerical solver of the equation f(x) = 0
    :param f: Function, left side of equation f(x) = 0 to solve
    :param g: Function, derivative of f
    :param x0: Float, initial guess
    :param e: Float, tolerable error
    :param N: Integer, maximal steps
    :return:
    """
    step = 1
    flag = 1
    condition = True
    while condition:
        if np.all(g(x0) == 0.0):
            print('Divide by zero error!')
            break
        x1 = x0 - f(x0) / g(x0)
        x0 = x1
        step = step + 1
        if step > N:
            flag = 0
            break
        condition = np.any(abs(f(x1)) > e)
    if flag == 1:
        return x1
    else:
        print('\nNot Convergent.')


def backward_euler1(y0, t0, t1, h):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    g = G
    m = mass
    N = int(np.ceil((t1 - t0) / h))
    v = np.zeros((len(y0), N + 1, 2))
    y = np.zeros((len(y0), N + 1, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    t = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    for k in range(1, N + 1):

        for i in range(len(y0)):
            t = t + h

            def fixpoint(x):
                terms = []
                for j in range(len(x)):
                    if j != i:
                        term = (-g * m[0] * m[1] * (x[i, :] - x[j, :]) / np.linalg.norm(x[i, :] - x[j, :]) ** 3) / m[i]
                        terms.append(term)

                return y[i, k - 1, 0] + h * (terms[0] + terms[1]) - x[i, :]

            def fixpoint_deriv(x):
                terms = []
                for j in range(len(x)):
                    if j != i:
                        term = (g * m[0] * m[1] * (2*x[i, :]**2 - 3*x[i, :]*x[j, :] + x[j, :]**2) / np.linalg.norm(x[i, :] - x[j, :]) ** (5/2)) / m[i]
                        terms.append(term)
                return h * (terms[0] + terms[1]) - 1

            y[i, k, :] = newton_raphson(fixpoint, fixpoint_deriv, y[:, k - 1, :], 0.0001, 20)[i, :]
            t_list[k] = t
    return y, t_list


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
    v = np.zeros((len(y0), N + 1, 2))
    y = np.zeros((len(y0), N + 1, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    for k in range(N):
        for i in range(len(y0)):
            print(y[:, k, :])
            k1 = f(t, y[:, k, :])[i]
            k2 = f(t + 0.5 * h, (y[:, k, :] + h * k1 / 2))[i]
            k3 = f(t + h, (y[:, k, :] + h * (-k1 + 2 * k2)))[i]
            v[i, k + 1] = v[i, k, :] + h / 6 * (k1 + 4 * k2 + k3)
            k12 = v[i, k, :]
            k22 = v[i, k, :] + h * k1 / 2
            k32 = v[i, k, :] + h * (-k1 + 2 * k2)
            y[i, k + 1] = y[i, k, :] + h / 6 * (k12 + 4 * k22 + k32)
            t = t + h
            t_list[k + 1] = t
    return y, v, t_list


y_vec, v_vec, t_vec = runge_kutta(f, inits7, t_start, t_stop, 0.01)
print(v_vec)

