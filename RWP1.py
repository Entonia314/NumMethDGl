import numpy as np
from scipy import sparse
from bokeh.plotting import figure, show
from math import log
from bokeh.layouts import row, column, gridplot
import matplotlib.pyplot as plt

ua = 0
ub = 2
N = [10, 20, 40, 80, 160, 320]
gitter = np.linspace(ua, ub, N[0]).transpose()


def f(x):
    return 1


def y_analytical(x):
    return -0.5 * x ** 2 + x


def finite_difference_BVP(f, gitter, ua, ub):
    n = len(gitter) - 1
    dx = (ub - ua) / n
    x = gitter

    b = np.zeros((n + 1, 1)).ravel()
    b[1:n] = f(x[1:n])

    main_diag = -2 * np.ones((n + 1, 1)).ravel()
    off_diag = 1 * np.ones((n, 1)).ravel()
    a = main_diag.shape[0]
    diagonals = [main_diag, off_diag, off_diag]
    A = sparse.diags(diagonals, [0, -1, 1], shape=(a, a)).toarray()
    A[0, 0] = 1
    A[0, 1] = 0
    A[n, n] = 1
    A[n, n - 1] = 0
    A = A * (1 / dx) ** 2
    print(A)
    print(b)

    u = -np.linalg.solve(A, b)
    return u


def finite_difference_BVP2(f, gitter, ua, ub):
    n = len(gitter) - 1
    dx = (ub - ua) / n
    x = gitter

    b = np.zeros((n-1, 1)).ravel()
    b[:] = f(x[:])

    main_diag = -2 * np.ones((n-1, 1)).ravel()
    off_diag = 1 * np.ones((n-2, 1)).ravel()
    a = main_diag.shape[0]
    diags = [main_diag, off_diag, off_diag]
    A = sparse.diags(diags, [0, -1, 1], shape=(a, a)).toarray()
    A = A * (1 / dx) ** 2

    u = -np.linalg.solve(A, b)
    u = [0] + u + [0]
    return u


xf = np.linspace(ua, ub, 1001)
y_exact = -0.5 * xf ** 2 + xf

p = figure(title="Finite Differenzen für Randwertproblem: -u'' = 1", x_axis_label='x', y_axis_label='u(x)')
p.line(xf, y_exact, legend_label='Exakte Lösung', line_width=2, line_color='red')
colours = ['green', 'purple', 'blue', 'yellow', 'pink', 'cyan', 'brown']

u_N = []
x_N = []
error_N = []
h_N = []
i = 0
for n in N:
    gitter = np.linspace(ua, ub, n + 1).transpose()
    y = finite_difference_BVP2(f, gitter, ua, ub)
    x_N.append(gitter)
    u_N.append(y)
    h_N.append((ub - ua) / n)
    p.line(gitter[1:-1], y, legend_label=str("N = " + str(n)), line_width=2, line_color=colours[i])
    i += 1
    errors = []
    for j in range(n-1):
        error = abs(y_analytical(gitter[j])-y[j])
        errors.append(error)
    error_N.append(max(errors))

print('Maximaler Fehler für N={10,20,40,80,160,320}: '+str(error_N))

conv_order_list = []
for n in range(-1, len(N)-2):
    conv_order = log(error_N[n+1]/error_N[n+2])/log(h_N[n+1]/h_N[n+2])
    conv_order_list.append(conv_order)

print('Konvergenzordnung: '+str(conv_order_list))

show(p)
