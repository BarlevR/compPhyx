'''
compPhyx
Directory: examples/ODE

Code: Radioactive decay solved with Euler, RK4, and RK45.

dy/dt = -y,  y(0) = 1  →  y(t) = e^{-t}

Author: Barlev Raymond
'''

import numpy as np
import matplotlib.pyplot as plt
import compPhyx.logo as logo
from compPhyx.timestepping import Euler, RK4, RK45

print(logo.art)

def f(t, y):
    return -y

t0   = 0
y0   = 1.0
nmax = 200
h    = 0.05

sol_euler = Euler(f, t0, y0, nmax, h).solve()
sol_rk4   = RK4(f, t0, y0, nmax, h).solve()
sol_rk45  = RK45(f, t0, y0, nmax, h).solve()

t_analytical = np.linspace(t0, t0 + nmax * h, 300)
y_analytical = np.exp(-t_analytical)

plt.figure(1)
plt.title('Radioactive decay — solutions')
plt.xlabel('t')
plt.ylabel('y')
plt.plot(t_analytical, y_analytical, 'k-', label='Analytical')
plt.plot(sol_euler[0], sol_euler[1], label='Euler')
plt.plot(sol_rk4[0],   sol_rk4[1],   label='RK4')
plt.plot(sol_rk45[0],  sol_rk45[1],  label='RK45')
plt.legend()

t_grid  = sol_euler[0]
y_exact = np.exp(-t_grid)

plt.figure(2)
plt.title('Radioactive decay — absolute error')
plt.xlabel('t')
plt.ylabel('|error|')
plt.semilogy(t_grid, np.abs(sol_euler[1] - y_exact), label='Euler')
plt.semilogy(t_grid, np.abs(sol_rk4[1]   - y_exact), label='RK4')
plt.semilogy(t_grid, np.abs(sol_rk45[1]  - y_exact), label='RK45')
plt.legend()

plt.show()
