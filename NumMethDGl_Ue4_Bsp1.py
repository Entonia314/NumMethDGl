import numpy as np
from scipy import sparse
from bokeh.plotting import figure, show
from math import log

# Global parameters, given from the exercise
a = 0
b = 2
ua = 0
ub = 0
N = [10, 20, 40, 80, 160, 320]


# Function from the ODE -u'' = 1
def f(x):
    return 1


# Analytical solution of the ODE
def y_analytical(x):
    return -0.5 * x ** 2 + x


# Implementation of the finite elements method for our ODE
def finite_elements_BVP(gitter, ua, ub):
    n = len(gitter) - 1
    a = gitter[0]
    b = gitter[-1]
    h = (b - a) / (n+1)
    x = gitter

    # Define the right side of the equation as integral of f(x)*phi(x) between a and b for every j
    b_vector = np.zeros((n-1, 1)).ravel()
    for j in range(1, n):
        b_vector[j-1] = (x[j]**2 - x[j-1]*x[j] + x[j-1]**2 / 2 + x[j+1]**2 / 2 - x[j+1]*x[j]) / h

    # Addition to the right side of the equation for boundary values that are not zero
    r = np.zeros((n-1, 1)).ravel()
    r[0] = ua
    r[-1] = ub
    r = r * (1 / h)

    b_vector = b_vector + r

    # Define the Matrix A of the left side of the equation
    main_diag = 2 * np.ones((n-1, 1)).ravel()
    off_diag = -1 * np.ones((n-2, 1)).ravel()
    a = main_diag.shape[0]
    diags = [main_diag, off_diag, off_diag]
    A = sparse.diags(diags, [0, -1, 1], shape=(a, a)).toarray()     # Sparse matrix for less computational effort
    A = A * (1 / h)

    # Solve the equation
    u = np.linalg.solve(A, b_vector)
    return u


# Exact solution for the plot
xf = np.linspace(a, b, 1001)
y_exact = -0.5 * xf ** 2 + xf

# Plot with Bokeh
p = figure(title="Finite Elemente Methode für Randwertproblem: -u''(x) = 1, u(0)=0, u(2)=0", x_axis_label='x', y_axis_label='u(x)')
p.line(xf, y_exact, legend_label='Exakte Lösung', line_width=2, line_color='red')
colours = ['green', 'purple', 'blue', 'yellow', 'pink', 'cyan', 'brown']

u_N = []    # List for solutions for every N
x_N = []    # List for grid for every N
error_N = []    # List for error for every N in the maximum norm
h_N = []    # List for step size h for every N
i = 0
for n in N:
    gitter = np.linspace(a, b, n + 1).transpose()       # Generates grid for every N
    y = finite_elements_BVP(gitter, ua, ub)     # Solves the equation for every N
    x_N.append(gitter)
    u_N.append(y)
    h_N.append((b - a) / n)
    p.line(gitter, np.concatenate([[ua], y, [ub]]), legend_label=str("N = " + str(n)), line_width=2,
           line_color=colours[i])       # Adds line to plot
    i += 1
    # Gets the maximal error for every N
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
