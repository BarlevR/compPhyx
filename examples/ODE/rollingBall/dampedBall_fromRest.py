'''
compPhyx
Directory: examples/ODE/rollingBall

Code: Ball in a parabolic bowl, released from rest.

Solvers compared:
    scipy.integrate.solve_ivp  (RK45)  — reference
    compPhyx.applications.RollingBall  — METHOD selectable below

Author: Barlev Raymond
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import compPhyx.logo as logo
from compPhyx.applications import RollingBall

print(logo.art)

# --- Pick compPhyx solver: 'Euler', 'RK4', 'RK45' ---
METHOD = 'RK45'

# --- Physical parameters ---
mass      = 1.0   # mass
curvature = 1.0   # bowl curvature  (potential U = curvature*(x^2 + y^2))
damping   = 0.1   # damping coefficient

# --- Initial conditions ---
x0  =  1.0  # initial x position
y0  = -0.5  # initial y position
vx0 =  0.0  # initial x velocity
vy0 =  0.0  # initial y velocity  (released from rest)

# --- Time span ---
tStart = 0.0
tEnd   = 100.0

# --- Problem setup ---
problem = RollingBall(m=mass, U0=curvature, xi=damping, r0=[x0, y0, vx0, vy0])
t_eval  = np.linspace(tStart, tEnd, 1001)

# --- scipy solve (reference) ---
sol_scipy = integrate.solve_ivp(problem.f, [tStart, tEnd], problem.r0,
                                method='RK45', t_eval=t_eval)

# --- compPhyx solve ---
sol_cp = problem.solve(method=METHOD, t_span=[tStart, tEnd], t_eval=t_eval)

# --- Plot: time series ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Damped ball in a bowl — released from rest')

axes[0].set_xlabel('t'); axes[0].set_ylabel('x(t)')
axes[0].plot(sol_scipy.t, sol_scipy.y[0], 'b-',  label='scipy RK45')
axes[0].plot(sol_cp.t,    sol_cp.y[0],    'r--', label=f'compPhyx {METHOD}', lw=1.5)
axes[0].legend()

axes[1].set_xlabel('t'); axes[1].set_ylabel('y(t)')
axes[1].plot(sol_scipy.t, sol_scipy.y[1], 'b-',  label='scipy RK45')
axes[1].plot(sol_cp.t,    sol_cp.y[1],    'r--', label=f'compPhyx {METHOD}', lw=1.5)
axes[1].legend()

plt.tight_layout()

# --- Plot: xy trajectory ---
fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.set_aspect('equal')
ax2.set_xlabel('x'); ax2.set_ylabel('y')
ax2.set_title('xy trajectory')
ax2.plot(sol_scipy.y[0], sol_scipy.y[1], 'b-',  label='scipy RK45')
ax2.plot(sol_cp.y[0],    sol_cp.y[1],    'r--', label=f'compPhyx {METHOD}', lw=1.5)
ax2.legend()

plt.show()
