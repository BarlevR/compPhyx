'''
compPhyx
Directory: examples/ODE/lorenz

Code: Lorenz system — sensitivity to initial conditions (chaos).
      Two trajectories with a small perturbation in x0 diverge exponentially.

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

# --- Physical parameters ---
a = 10.0        # Prandtl number
b = 50.0        # Rayleigh number  (b > 24.74 → chaotic)
c = 8.0 / 3.0  # geometric factor

# --- Initial conditions (two nearby trajectories) ---
x0  = 1.0   # initial x
y0  = 0.0   # initial y
z0  = 0.0   # initial z
dx0 = 0.1   # perturbation in x  (10% of x0)

# --- Time span ---
tStart = 0.0
tEnd   = 50.0

# --- Problem setup ---
problem  = LorenzSystem(a=a, b=b, c=c, r0=[x0,       y0, z0])
problem2 = LorenzSystem(a=a, b=b, c=c, r0=[x0 + dx0, y0, z0])
t_eval   = np.linspace(tStart, tEnd, 5001)

# --- scipy solve (reference) ---
sol1_scipy = integrate.solve_ivp(problem.f,  [tStart, tEnd], problem.r0,
                                 method='RK45', t_eval=t_eval)
sol2_scipy = integrate.solve_ivp(problem2.f, [tStart, tEnd], problem2.r0,
                                 method='RK45', t_eval=t_eval)

# --- compPhyx solve ---
sol1_cp = solve_ode(problem.f,  [tStart, tEnd], problem.r0,
                    method=METHOD, t_eval=t_eval)
sol2_cp = solve_ode(problem2.f, [tStart, tEnd], problem2.r0,
                    method=METHOD, t_eval=t_eval)

# --- Plot: 3D attractor — both trajectories, both solvers ---
fig = plt.figure(figsize=(14, 6))
fig.suptitle('Lorenz — sensitivity to initial conditions')

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_title('scipy RK45')
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
ax1.plot3D(sol1_scipy.y[0], sol1_scipy.y[1], sol1_scipy.y[2], 'b', lw=0.5, label='r0')
ax1.plot3D(sol2_scipy.y[0], sol2_scipy.y[1], sol2_scipy.y[2], 'r', lw=0.5, label='r0 + δ')
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_title(f'compPhyx {METHOD}')
ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
ax2.plot3D(sol1_cp.y[0], sol1_cp.y[1], sol1_cp.y[2], 'b', lw=0.5, label='r0')
ax2.plot3D(sol2_cp.y[0], sol2_cp.y[1], sol2_cp.y[2], 'r', lw=0.5, label='r0 + δ')
ax2.legend()

plt.tight_layout()

# --- Plot: x(t) divergence ---
fig2, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].set_title('x(t) — two trajectories')
axes[0].set_ylabel('x(t)')
axes[0].plot(sol1_scipy.t, sol1_scipy.y[0], 'b-',  lw=0.8, label='scipy  r0')
axes[0].plot(sol1_scipy.t, sol2_scipy.y[0], 'r-',  lw=0.8, label='scipy  r0+δ')
axes[0].plot(sol1_cp.t,    sol1_cp.y[0],    'b--', lw=0.8, label=f'compPhyx {METHOD}  r0')
axes[0].plot(sol1_cp.t,    sol2_cp.y[0],    'r--', lw=0.8, label=f'compPhyx {METHOD}  r0+δ')
axes[0].legend(fontsize=8)

axes[1].set_title('|x1(t) − x2(t)| — divergence of nearby trajectories')
axes[1].set_xlabel('t'); axes[1].set_ylabel('|Δx(t)|')
axes[1].semilogy(sol1_scipy.t, np.abs(sol1_scipy.y[0] - sol2_scipy.y[0]),
                 'b-',  lw=0.8, label='scipy RK45')
axes[1].semilogy(sol1_cp.t,    np.abs(sol1_cp.y[0] - sol2_cp.y[0]),
                 'r--', lw=0.8, label=f'compPhyx {METHOD}')
axes[1].legend()

plt.tight_layout()
plt.show()
