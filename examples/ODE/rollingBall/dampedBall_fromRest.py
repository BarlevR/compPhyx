'''
compPhyx
Directory: examples/ODE/rollingBall

Code: Ball in a parabolic bowl, released from rest.

Solvers compared:
    scipy.integrate.solve_ivp  (RK45)
    compPhyx.timestepping.RK45

Author: Barlev Raymond
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import compPhyx.logo as logo
from compPhyx.applications import RollingBall
from compPhyx.timestepping import RK45

print(logo.art)

# --- Problem setup ---
problem = RollingBall(m=1.0, U0=1.0, xi=0.1,
                      r0=[1.0, -0.5, 0.0, 0.0])

tStart, tEnd = 0.0, 100.0

# --- scipy solve ---
t_eval    = np.linspace(tStart, tEnd, 1001)
sol_scipy = integrate.solve_ivp(problem.f, [tStart, tEnd], problem.r0,
                                method='RK45', t_eval=t_eval)

# --- compPhyx solve ---
nmax   = 1000
h      = (tEnd - tStart) / nmax
sol_cp = RK45(problem.f, t0=tStart, y0=problem.r0, nmax=nmax, h=h).solve()

# --- Plot: time series ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Damped ball in a bowl — released from rest')

axes[0].set_xlabel('t'); axes[0].set_ylabel('x(t)')
axes[0].plot(sol_scipy.t, sol_scipy.y[0], 'b-',  label='scipy RK45')
axes[0].plot(sol_cp[0],   sol_cp[1],      'r--', label='compPhyx RK45', lw=1.5)
axes[0].legend()

axes[1].set_xlabel('t'); axes[1].set_ylabel('y(t)')
axes[1].plot(sol_scipy.t, sol_scipy.y[1], 'b-',  label='scipy RK45')
axes[1].plot(sol_cp[0],   sol_cp[2],      'r--', label='compPhyx RK45', lw=1.5)
axes[1].legend()

plt.tight_layout()

# --- Plot: xy trajectory ---
fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.set_aspect('equal')
ax2.set_xlabel('x'); ax2.set_ylabel('y')
ax2.set_title('xy trajectory')
ax2.plot(sol_scipy.y[0], sol_scipy.y[1], 'b-',  label='scipy RK45')
ax2.plot(sol_cp[1],      sol_cp[2],      'r--', label='compPhyx RK45', lw=1.5)
ax2.legend()

plt.show()
