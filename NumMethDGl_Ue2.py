import numpy as np
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from math import log
from sympy import symbols, solve

# Global parameters
t0 = 0
t1 = 10
y0 = 1
h1 = 0.1
h2 = 0.01


# Function of the right side of the ODE
def f(t, y):
    dy = [0] * len(y)
    dy[0] = -2 * t * y[0] ** 2
    return dy


# Exact solution of the ODE
def exakt(t):
    return 1 / (t**2 + 1)

# Implementation of some numerical methods to solve the ODE


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
        if g(x0) == 0.0:
            print('Divide by zero error!')
            break
        x1 = x0 - f(x0) / g(x0)
        x0 = x1
        step = step + 1
        if step > N:
            flag = 0
            break
        condition = abs(f(x1)) > e
    if flag == 1:
        return x1
    else:
        print('\nNot Convergent.')


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
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    y = np.zeros((len(y0), N + 1))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    y[:, 0] = y0
    for k in range(1, N + 1):
        for i in range(len(y0)):
            t = t + h

            def equation(x):
                return y[i, k-1] + h * (-2) * t * x ** 2 - x

            def equation_diff(x):
                return h * (-4) * t * x - 1

            y[i, k] = newton_raphson(equation, equation_diff, y[i, k-1], 0.0001, 10)
            t_list[k] = t
    return y, t_list


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
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    y = np.zeros((len(y0), N + 1))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    y[:, 0] = y0
    for k in range(1, N + 1):
        for i in range(len(y0)):
            t = t + h

            def equation(x):
                return y[i, k-1] + h/2 * (((-2) * t * y[i, k-1] ** 2) + ((-2) * t * x ** 2)) - x

            def equation_diff(x):
                return h/2 * (-4) * t * x - 1

            y[i, k] = newton_raphson(equation, equation_diff, y[i, k-1], 0.0001, 10)
            t_list[k] = t
    return y, t_list


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
            k2 = f(t + 0.5*h, add(y[:, k], h * k1 / 2))[i]
            k3 = f(t + h, add(y[:, k], h * (-k1 + 2 * k2)))[i]
            y[i, k + 1] = y[i, k] + h / 6 * (k1 + 4 * k2 + k3)
            t = t + h
            t_list[k + 1] = t
    return y, t_list


def adams_bashforth_3(f, y0, t0, t1, h, g):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :param g: function to calculate starting values
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
    y_start, s = g
    y[:, 1] = y_start[:, 1]
    y[:, 2] = y_start[:, 2]
    for k in range(2, N):
        for i in range(len(y0)):
            y[i, k + 1] = y[i, k] + h * ((23 / 12 * f(t, y[:, k])[i]) - (16 / 12 * f(t - h, y[:, k - 1])[i]) + (
                        5 / 12 * f(t - 2 * h, y[:, k - 2])[i]))
            t = t + h
            t_list[k + 1] = t
    return y, t_list


y11, s11 = forward_euler(f, [y0], t0, t1, h1)
y21, s21 = backward_euler(f, [y0], t0, t1, h1)
y31, s31 = crank_nicolson(f, [y0], t0, t1, h1)
y41, s41 = runge_kutta(f, [y0], t0, t1, h1)
y511, s511 = adams_bashforth_3(f, [y0], t0, t1, h1, forward_euler(f, [y0], t0, t1, h1))
y512, s512 = adams_bashforth_3(f, [y0], t0, t1, h1, runge_kutta(f, [y0], t0, t1, h1))

y12, s12 = forward_euler(f, [y0], t0, t1, h2)
y22, s22 = backward_euler(f, [y0], t0, t1, h2)
y32, s32 = crank_nicolson(f, [y0], t0, t1, h2)
y42, s42 = runge_kutta(f, [y0], t0, t1, h2)
y521, s521 = adams_bashforth_3(f, [y0], t0, t1, h2, forward_euler(f, [y0], t0, t1, h2))
y522, s522 = adams_bashforth_3(f, [y0], t0, t1, h2, runge_kutta(f, [y0], t0, t1, h2))


# Exact solution for plot

exakt_y1 = []
for t in s11:
    exakt_y1.append(exakt(t))

exakt_y2 = []
for t in s12:
    exakt_y2.append(exakt(t))


# Printing the solutions at t=5 and the order of convergence
print('Lösungen bei t=5: Exakt: '+str(exakt(10)))
print('Explizites Euler-Verfahren - h=0.1: '+str(y11[0][-1])+', h=0.01: '+str(y12[0][-1])+
      ', Fehlerordnung: '+str(log(abs((y11[0][-1]-exakt(10))/(y12[0][-1]-exakt(10))))/log(h1/h2)))
print('Implizites Euler-Verfahren - h=0.1: '+str(y21[0][-1])+', h=0.01: '+str(y22[0][-1])+
      ', Fehlerordnung: '+str(log(abs(y21[0][-1]-exakt(10))/abs(y22[0][-1]-exakt(10)))/log(h1/h2)))
print('Cranc-Nicolson-Verfahren - h=0.1: '+str(y31[0][-1])+', h=0.01: '+str(y32[0][-1])+
      ', Fehlerordnung: '+str(log(abs(y31[0][-1]-exakt(10))/abs(y32[0][-1]-exakt(10)))/log(h1/h2)))
print('Runge-Kutta-Verfahren - h=0.1: '+str(y41[0][-1])+', h=0.01: '+str(y42[0][-1])+
      ', Fehlerordnung: '+str(log(abs((y41[0][-1]-exakt(10))/(y42[0][-1]-exakt(10))))/log(h1/h2)))
print('Adams-Bashforth-Verfahren (Startwerte: Expl. Euler) - h=0.1: '+str(y511[0][-1])+', h=0.01: '+str(y521[0][-1])+
      ', Fehlerordnung: '+str(log(abs((y511[0][-1]-exakt(10))/(y521[0][-1]-exakt(10))))/log(h1/h2)))
print('Adams-Bashforth-Verfahren (Startwerte: Runge-Kutta) - h=0.1: '+str(y512[0][-1])+', h=0.01: '+str(y522[0][-1])+
      ', Fehlerordnung: '+str(log(abs((y512[0][-1]-exakt(10))/(y522[0][-1]-exakt(10))))/log(h1/h2)))


# Generating the plots

p = figure(title="h = 0.1", x_axis_label='t', y_axis_label='y(t)')
p.line(s11, y11[0], legend_label="Forward Euler", line_width=2, line_color="green")
p.line(s21, y21[0], legend_label="Backward Euler", line_width=2, line_color="blue")
p.line(s31, y31[0], legend_label="Cranc-Nicolson", line_width=2, line_color="purple")
p.line(s41, y41[0], legend_label="Runge-Kutta Ord. 3", line_width=2, line_color="red")
p.line(s512, y512[0], legend_label="Adam-Bashforth Ord. 3", line_width=2, line_color="cyan")

q = figure(title="h = 0.01", x_axis_label='t', y_axis_label='y(t)')
q.line(s12, y12[0], legend_label="Forward Euler", line_width=2, line_color="green")
q.line(s22, y22[0], legend_label="Backward Euler", line_width=2, line_color="blue")
q.line(s32, y32[0], legend_label="Cranc-Nicolson", line_width=2, line_color="purple")
q.line(s42, y42[0], legend_label="Runge-Kutta Ord. 3", line_width=2, line_color="red")
q.line(s522, y522[0], legend_label="Adam-Bashforth Ord. 3", line_width=2, line_color="cyan")

p1 = figure(title="Forward Euler h = 0.1", x_axis_label='t', y_axis_label='y(t)')
p1.line(s11, y11[0], legend_label="Forward Euler", line_width=2, line_color="green")
p1.line(s11, exakt_y1, legend_label="Exakt", line_width=2, line_color="pink")

q1 = figure(title="Forward Euler h = 0.01", x_axis_label='t', y_axis_label='y(t)')
q1.line(s12, y12[0], legend_label="Forward Euler", line_width=2, line_color="green")
q1.line(s12, exakt_y2, legend_label="Exakt", line_width=2, line_color="pink")

p2 = figure(title="Backward Euler h = 0.1", x_axis_label='t', y_axis_label='y(t)')
p2.line(s11, y21[0], legend_label="Backward Euler", line_width=2, line_color="blue")
p2.line(s11, exakt_y1, legend_label="Exakt", line_width=2, line_color="pink")

q2 = figure(title="Backward Euler h = 0.01", x_axis_label='t', y_axis_label='y(t)')
q2.line(s12, y22[0], legend_label="Backward Euler", line_width=2, line_color="blue")
q2.line(s12, exakt_y2, legend_label="Exakt", line_width=2, line_color="pink")

p3 = figure(title="Crank-Nicolson h = 0.1", x_axis_label='t', y_axis_label='y(t)')
p3.line(s11, y31[0], legend_label="Crank-Nicolson", line_width=2, line_color="purple")
p3.line(s11, exakt_y1, legend_label="Exakt", line_width=2, line_color="pink")

q3 = figure(title="Crank-Nicolson h = 0.01", x_axis_label='t', y_axis_label='y(t)')
q3.line(s12, y32[0], legend_label="Crank-Nicolson", line_width=2, line_color="purple")
q3.line(s12, exakt_y2, legend_label="Exakt", line_width=2, line_color="pink")

p4 = figure(title="Runge-Kutta Ord. 3, h = 0.1", x_axis_label='t', y_axis_label='y(t)')
p4.line(s11, y41[0], legend_label="Runge-Kutta Ord. 3", line_width=2, line_color="red")
p4.line(s11, exakt_y1, legend_label="Exakt", line_width=2, line_color="pink")

q4 = figure(title="Runge-Kutta Ord. 3, h = 0.01", x_axis_label='t', y_axis_label='y(t)')
q4.line(s12, y42[0], legend_label="Runge-Kutta Ord. 3", line_width=2, line_color="red")
q4.line(s12, exakt_y2, legend_label="Exakt", line_width=2, line_color="pink")

p51 = figure(title="Adams-Bashforth mit Forward Euler-Startwerten, h = 0.1", x_axis_label='t', y_axis_label='y(t)')
p51.line(s11, y511[0], legend_label="Adams-Bashforth", line_width=2, line_color="cyan")
p51.line(s11, exakt_y1, legend_label="Exakt", line_width=2, line_color="pink")

q51 = figure(title="Adams-Bashforth mit Forward Euler-Startwerten, h = 0.01", x_axis_label='t', y_axis_label='y(t)')
q51.line(s12, y521[0], legend_label="Adams-Bashforth", line_width=2, line_color="cyan")
q51.line(s12, exakt_y2, legend_label="Exakt", line_width=2, line_color="pink")

p52 = figure(title="Adams-Bashforth mit Runge-Kutta-Startwerten, h = 0.1", x_axis_label='t', y_axis_label='y(t)')
p52.line(s11, y512[0], legend_label="Adams-Bashforth", line_width=2, line_color="cyan")
p52.line(s11, exakt_y1, legend_label="Exakt", line_width=2, line_color="pink")

q52 = figure(title="Adams-Bashforth mit Runge-Kutta-Startwerten, h = 0.01", x_axis_label='t', y_axis_label='y(t)')
q52.line(s12, y522[0], legend_label="Adams-Bashforth", line_width=2, line_color="cyan")
q52.line(s12, exakt_y2, legend_label="Exakt", line_width=2, line_color="pink")

grid = gridplot([[p, p1, p2, p3, p4, p51, p52], [q, q1, q2, q3, q4, q51, q52]])
show(grid)
