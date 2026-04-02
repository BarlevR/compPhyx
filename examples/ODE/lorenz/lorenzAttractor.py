'''
compPhyx
Directory: examples/ODE/lorenz

Code: The Lorenz attractor — chaotic strange attractor.
      For b > 24.74 the system is chaotic.

Solvers compared:
    scipy.integrate.solve_ivp  (RK45)  — reference
    compPhyx.timestepping.solve_ode    — METHOD selectable below

Author: Barlev Raymond
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import compPhyx.logo as logo
from compPhyx.applications import LorenzSystem
from compPhyx.timestepping import solve_ode

print(logo.art)

# --- Pick compPhyx solver: 'Euler', 'RK4', 'RK45' ---
METHOD = 'RK45'

# --- Problem setup ---
problem = LorenzSystem(a=10.0, b=50.0, c=8.0/3.0,
                       r0=[1.0, 0.0, 0.0])

tStart, tEnd = 0.0, 50.0
t_eval = np.linspace(tStart, tEnd, 5001)

# --- scipy solve (reference) ---
sol_scipy = integrate.solve_ivp(problem.f, [tStart, tEnd], problem.r0,
                                method='RK45', t_eval=t_eval)

# --- compPhyx solve ---
sol_cp = solve_ode(problem.f, [tStart, tEnd], problem.r0,
                   method=METHOD, t_eval=t_eval)

# --- Plot: 3D attractor ---
fig = plt.figure(figsize=(14, 6))
fig.suptitle('Lorenz attractor  (a={}, b={}, c={:.3f})'.format(
    problem.a, problem.b, problem.c))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_title('scipy RK45')
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
ax1.plot3D(sol_scipy.y[0], sol_scipy.y[1], sol_scipy.y[2], lw=0.5)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_title(f'compPhyx {METHOD}')
ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
ax2.plot3D(sol_cp.y[0], sol_cp.y[1], sol_cp.y[2], lw=0.5)

plt.tight_layout()

# --- Plot: x(t) time series ---
fig2, ax = plt.subplots(figsize=(10, 4))
ax.set_title('Lorenz — x(t)')
ax.set_xlabel('t'); ax.set_ylabel('x(t)')
ax.plot(sol_scipy.t, sol_scipy.y[0], 'b-',  lw=0.8, label='scipy RK45')
ax.plot(sol_cp.t,    sol_cp.y[0],    'r--', lw=0.8, label=f'compPhyx {METHOD}')
ax.legend()
plt.tight_layout()

plt.show()
