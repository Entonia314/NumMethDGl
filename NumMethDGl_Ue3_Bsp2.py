import numpy as np
from scipy import sparse
from bokeh.plotting import figure, show
from math import log

# Global parameters, given from the exercise
a = 0
b = 2
ua = 1
ub = 2
N = [10, 20, 40, 80, 160, 320]


# Function from the ODE -u'' = x
def f(x):
    return x


# Analytical solution of the ODE
def y_analytical(x):
    return -1/6 * x**3 + 7/6 * x + 1


# Implementation of the finite differences method for our ODE
def finite_difference_BVP(f, gitter, ua, ub):
    n = len(gitter) - 1
    a = gitter[0]
    b = gitter[-1]
    dx = (b - a) / n
    x = gitter

    # Define the right side of the equation
    b_vector = np.zeros((n-1, 1)).ravel()
    b_vector[:] = f(x[1:n])

    # Addition to the right side of the equation for boundary values that are not zero
    r = np.zeros((n-1, 1)).ravel()
    r[0] = ua
    r[-1] = ub
    r = r * (1 / dx) ** 2

    b_vector = b_vector + r

    # Define the Matrix A of the left side of the equation
    main_diag = -2 * np.ones((n-1, 1)).ravel()
    off_diag = 1 * np.ones((n-2, 1)).ravel()
    a = main_diag.shape[0]
    diags = [main_diag, off_diag, off_diag]
    A = sparse.diags(diags, [0, -1, 1], shape=(a, a)).toarray()
    A = A * (1 / dx) ** 2

    # Solve the equation
    u = -np.linalg.solve(A, b_vector)
    return u


# Exact solution for the plot
xf = np.linspace(a, b, 1001)
y_exact = -1/6 * xf**3 + 7/6 * xf + 1

# Plot with Bokeh
p = figure(title="Finite Differenzen für Randwertproblem: -u''(x) = x, u(0)=1, u(2)=2", x_axis_label='x', y_axis_label='u(x)')
p.line(xf, y_exact, legend_label='Exakte Lösung', line_width=2, line_color='red')
colours = ['green', 'purple', 'blue', 'yellow', 'pink', 'cyan', 'brown']

u_N = []  # List for solutions for every N
x_N = []  # List for grid for every N
error_N = []  # List for error for every N in the maximum norm
h_N = []  # List for step size h for every N
i = 0
for n in N:
    gitter = np.linspace(a, b, n + 1).transpose()  # Generates grid for every N
    y = finite_difference_BVP(f, gitter, ua, ub)  # Solves the equation for every N
    x_N.append(gitter)
    u_N.append(y)
    h_N.append((b - a) / n)
    p.line(gitter, np.concatenate([[ua], y, [ub]]), legend_label=str("N = " + str(n)), line_width=2,
           line_color=colours[i])   # Adds line to plot
    i += 1
    errors = []
    for j in range(n-1):
        error = abs(y_analytical(gitter[j])-y[j])
        errors.append(error)
    error_N.append(max(errors))

print('Maximaler Fehler für N={10,20,40,80,160,320}: '+str(error_N))

# Calculates the order of convergence for every two consecutive Ns
conv_order_list = []
for n in range(len(N)-1):
    conv_order = log(error_N[n]/error_N[n+1])/log(h_N[n]/h_N[n+1])
    conv_order_list.append(conv_order)

print('Konvergenzordnung: '+str(conv_order_list))

show(p)
