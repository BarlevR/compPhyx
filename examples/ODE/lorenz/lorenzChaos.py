'''
compPhyx
Directory: examples/ODE/lorenz

Code: Lorenz system — sensitivity to initial conditions (chaos).

      Two trajectories with nearly identical starting points diverge
      exponentially. This is the hallmark of a chaotic system.

      Initial conditions:
          Trajectory 1:  r0  = [x0,       0, 0]
          Trajectory 2:  r0b = [x0 * 1.1, 0, 0]   (10% perturbation in x)

Solvers compared:
    scipy.integrate.solve_ivp  (RK45)
    compPhyx.timestepping.RK45

Author: Barlev Raymond
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import compPhyx.logo as logo
from compPhyx.timestepping import RK45

print(logo.art)

# --- Lorenz parameters ---
a = 10.0
b = 50.0
c = 8.0 / 3.0

# --- ODE right-hand side ---
def f_lorenz(t, r, a, b, c):
    x, y, z = r
    return np.array([a*(y - x), x*(b - z) - y, x*y - c*z])

def f(t, r):
    return f_lorenz(t, r, a, b, c)

# --- Initial conditions ---
tStart = 0.0
tEnd   = 50.0
r0     = [1.0, 0.0, 0.0]
r0b    = [1.1, 0.0, 0.0]   # 10% perturbation

# --- scipy solve ---
t_eval = np.linspace(tStart, tEnd, 5001)
sol1_scipy = integrate.solve_ivp(f_lorenz, [tStart, tEnd], r0,
                                 method='RK45', t_eval=t_eval, args=(a, b, c))
sol2_scipy = integrate.solve_ivp(f_lorenz, [tStart, tEnd], r0b,
                                 method='RK45', t_eval=t_eval, args=(a, b, c))

# --- compPhyx solve ---
nmax    = 5000
h       = (tEnd - tStart) / nmax
sol1_cp = RK45(f, t0=tStart, y0=r0,  nmax=nmax, h=h).solve()
sol2_cp = RK45(f, t0=tStart, y0=r0b, nmax=nmax, h=h).solve()

# --- Plot: 3D attractor — both trajectories, both solvers ---
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_title('scipy RK45')
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
ax1.plot3D(sol1_scipy.y[0], sol1_scipy.y[1], sol1_scipy.y[2], 'b', lw=0.5, label='r0')
ax1.plot3D(sol2_scipy.y[0], sol2_scipy.y[1], sol2_scipy.y[2], 'r', lw=0.5, label='r0 + δ')
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_title('compPhyx RK45')
ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
ax2.plot3D(sol1_cp[1], sol1_cp[2], sol1_cp[3], 'b', lw=0.5, label='r0')
ax2.plot3D(sol2_cp[1], sol2_cp[2], sol2_cp[3], 'r', lw=0.5, label='r0 + δ')
ax2.legend()

fig.suptitle('Lorenz — sensitivity to initial conditions')
plt.tight_layout()

# --- Plot: x(t) divergence ---
fig2, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axes[0].set_title('x(t) — two trajectories')
axes[0].set_ylabel('x(t)')
axes[0].plot(sol1_scipy.t, sol1_scipy.y[0], 'b-',  lw=0.8, label='scipy  r0')
axes[0].plot(sol1_scipy.t, sol2_scipy.y[0], 'r-',  lw=0.8, label='scipy  r0+δ')
axes[0].plot(sol1_cp[0],   sol1_cp[1],      'b--', lw=0.8, label='compPhyx  r0')
axes[0].plot(sol1_cp[0],   sol2_cp[1],      'r--', lw=0.8, label='compPhyx  r0+δ')
axes[0].legend(fontsize=8)

axes[1].set_title('|x1(t) − x2(t)| — divergence of nearby trajectories')
axes[1].set_xlabel('t')
axes[1].set_ylabel('|Δx(t)|')
div_scipy = np.abs(sol1_scipy.y[0] - sol2_scipy.y[0])
div_cp    = np.abs(sol1_cp[1] - sol2_cp[1])
axes[1].semilogy(sol1_scipy.t, div_scipy, 'b-',  lw=0.8, label='scipy RK45')
axes[1].semilogy(sol1_cp[0],   div_cp,    'r--', lw=0.8, label='compPhyx RK45')
axes[1].legend()

plt.tight_layout()
plt.show()
