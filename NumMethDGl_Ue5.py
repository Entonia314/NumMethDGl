import numpy as np
from scipy import sparse
from bokeh.plotting import figure, show
from math import log

# Global parameters, given from the exercise
xmin = 0
xmax = 1
tmax = 2
dt = 0.009
N = [50, 400]


# Function from the ODE -u'' = x
def f(x):
    return np.sin(2*np.pi*x)


# Analytical solution of the ODE
def y_analytical(x, t):
    return np.sin(x+t)


# Implementation of the finite elements method for our ODE
def LaxFriedrich(xmin, xmax, N):
    dx = (xmax - xmin) / N
    x = np.arange(xmin-dx, xmax+2*dx, dx)
    u0 = f(x)
    u = u0.copy()
    unp1 = u0.copy()



    return u


# Plot with Bokeh
p = figure(title="Finite Elemente Methode für Randwertproblem: -u''(x) = x, u(0)=1, u(2)=2", x_axis_label='x',
           y_axis_label='u(x)')
p.line(xf, y_exact, legend_label='Exakte Lösung', line_width=2, line_color='red')
colours = ['green', 'purple', 'blue', 'yellow', 'pink', 'cyan', 'brown']

u_N = []  # List for solutions for every N
x_N = []  # List for grid for every N
error_N = []  # List for error for every N in the maximum norm
h_N = []  # List for step size h for every N
i = 0
for n in N:
    gitter = np.linspace(a, b, n + 1).transpose()  # Generates grid for every N
    y = LaxFriedrich(gitter, ua, ub)  # Solves the equation for every N
    x_N.append(gitter)
    u_N.append(y)
    h_N.append((b - a) / n)
    p.line(gitter, np.concatenate([[ua], y, [ub]]), legend_label=str("N = " + str(n)), line_width=2,
           line_color=colours[i])  # Adds line to plot
    i += 1
    errors = []
    for j in range(n - 1):
        error = abs(y_analytical(gitter[j]) - y[j])
        errors.append(error)
    error_N.append(max(errors))

print('Maximaler Fehler für N={10,20,40,80,160,320}: ' + str(error_N))

# Calculates the order of convergence for every two consecutive Ns
conv_order_list = []
for n in range(len(N) - 1):
    conv_order = log(error_N[n] / error_N[n + 1]) / log(h_N[n] / h_N[n + 1])
    conv_order_list.append(conv_order)

print('Konvergenzordnung: ' + str(conv_order_list))

show(p)
