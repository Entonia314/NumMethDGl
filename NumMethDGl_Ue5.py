import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot


# Global parameters, given from the exercise
xmin = 0
xmax = 1
tmax = 1
k = [0.01, 0.001]
N = [50, 400]

a = -1


# u(x,0) = sin(2*Pi*x)
def f(x):
    return np.sin(2*np.pi*x)


# Analytical solution of the Advection Equation
def u_analytical(x, t):
    return np.sin(2*np.pi*(x-a*t))


# Implementation of the Upwind method
def Upwind(xmin, xmax, N, dt):
    dx = (xmax - xmin) / N
    x = np.arange(xmin-dx, xmax+2*dx, dx)
    u0 = f(x)
    nsteps = round(tmax/dt)
    u = np.zeros((len(x), nsteps))
    u[:, 0] = u0

    tc = 0
    t = np.array([0])

    for i in range(nsteps-1):
        for j in range(N+2):
            u[j, i+1] = (-a * dt * (u[j+1, i] - u[j, i])) / dx + u[j, i]

        u[0, i+1] = u[N + 1, i+1]
        u[N + 2, i+1] = u[1, i+1]

        t = np.append(t, tc)
        tc += dt

    return u, x, t


# Implementation of the Lax-Friedrich method
def LaxFriedrich(xmin, xmax, N, dt):
    dx = (xmax - xmin) / N
    x = np.arange(xmin-dx, xmax+2*dx, dx)
    u0 = f(x)
    nsteps = round(tmax/dt)
    u = np.zeros((len(x), nsteps))
    u[:, 0] = u0

    tc = 0
    t = np.array([0])

    for i in range(nsteps-1):
        for j in range(N+2):
            u[j, i+1] = 0.5 * (u[j+1, i] + u[j-1, i]) - a * (dt / 2*dx) * (u[j+1, i] - u[j-1, i])

        u[0, i+1] = u[N + 1, i+1]
        u[N + 2, i+1] = u[1, i+1]

        t = np.append(t, tc)
        tc += dt

    return u, x, t


u50, x50, t50 = LaxFriedrich(xmin, xmax, 50, 0.009)
u400, x400, t400 = LaxFriedrich(xmin, xmax, 400, 0.002)

v50, y50, s50 = Upwind(xmin, xmax, 50, 0.009)
v400, y400, s400 = Upwind(xmin, xmax, 400, 0.002)

time1 = 0.5
time2 = 1.0

# Plot with Bokeh
pLF = figure(title=str("Lax-Friedrich-Methode bei t = " + str(time1)), x_axis_label='x',
             y_axis_label='u(x)')
pLF.line(x50, u_analytical(x50, t50[round(len(t50) * time1)]), legend_label='Exakte Lösung', line_width=2, line_color='red')
pLF.line(x50, u50[:, round(len(t50) * time1)], legend_label='n = 50, dt = 0.01', line_width=2, line_color='blue')
pLF.line(x400, u400[:, round(len(t400) * time1)], legend_label='n = 400, dt = 0.001', line_width=2, line_color='green')

qLF = figure(title=str("Lax-Friedrich-Methode bei t = " + str(time2)), x_axis_label='x',
             y_axis_label='u(x)')
qLF.line(x50, u_analytical(x50, t50[round(len(t50) * time2) - 1]), legend_label='Exakte Lösung', line_width=2, line_color='red')
qLF.line(x50, u50[:, round(len(t50) * time2) - 1], legend_label='n = 50, dt = 0.01', line_width=2, line_color='blue')
qLF.line(x400, u400[:, round(len(t400) * time2) - 1], legend_label='n = 400, dt = 0.001', line_width=2, line_color='green')

pUp = figure(title=str("Upwind-Methode bei t = " + str(time1)), x_axis_label='x',
             y_axis_label='u(x)')
pUp.line(y50, u_analytical(y50, s50[round(len(s50) * time1)]), legend_label='Exakte Lösung', line_width=2, line_color='red')
pUp.line(y50, v50[:, round(len(s50) * time1)], legend_label='n = 50, dt = 0.01', line_width=2, line_color='blue')
pUp.line(y400, v400[:, round(len(s400) * time1)], legend_label='n = 400, dt = 0.001', line_width=2, line_color='green')

qUp = figure(title=str("Upwind-Methode bei t = " + str(time2)), x_axis_label='x',
             y_axis_label='u(x)')
qUp.line(y50, u_analytical(y50, t50[round(len(t50) * time2) - 1]), legend_label='Exakte Lösung', line_width=2, line_color='red')
qUp.line(y50, v50[:, round(len(s50) * time2) - 1], legend_label='n = 50, dt = 0.01', line_width=2, line_color='blue')
qUp.line(y400, v400[:, round(len(s400) * time2) - 1], legend_label='n = 400, dt = 0.001', line_width=2, line_color='green')

grid = gridplot([[pLF, qLF], [pUp, qUp]])
show(grid)

print(t50[round(len(t50)*time1)], t400[round(len(t400)*time1)])
print(t50[round(len(t50)*time2)-1], t400[round(len(t400)*time2)-1])
