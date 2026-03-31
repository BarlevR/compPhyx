'''
compPhyx
Directory: examples/ODE/lorenz

Code: The Lorenz attractor — chaotic strange attractor.

      Equations:
          x' = a*(y - x)
          y' = x*(b - z) - y
          z' = x*y - c*z

      For b > 24.74 the system is chaotic and trajectories are attracted
      to the Lorenz butterfly (strange attractor).

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
b = 50.0      # > 24.74 → chaotic
c = 8.0 / 3.0

# --- ODE right-hand side ---
def f_lorenz(t, r, a, b, c):
    x, y, z = r
    return np.array([a*(y - x), x*(b - z) - y, x*y - c*z])

# Closure so our solver (no args support) gets a plain f(t, r)
def f(t, r):
    return f_lorenz(t, r, a, b, c)

# --- Initial conditions ---
tStart = 0.0
tEnd   = 50.0
r0     = [1.0, 0.0, 0.0]

# --- scipy solve ---
t_eval    = np.linspace(tStart, tEnd, 5001)
sol_scipy = integrate.solve_ivp(f_lorenz, [tStart, tEnd], r0,
                                method='RK45', t_eval=t_eval, args=(a, b, c))

# --- compPhyx solve ---
nmax   = 5000
h      = (tEnd - tStart) / nmax
sol_cp = RK45(f, t0=tStart, y0=r0, nmax=nmax, h=h).solve()
# sol_cp[0]=t, sol_cp[1]=x, sol_cp[2]=y, sol_cp[3]=z

# --- Plot: 3D attractor ---
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_title('scipy RK45')
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
ax1.plot3D(sol_scipy.y[0], sol_scipy.y[1], sol_scipy.y[2], linewidth=0.5)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_title('compPhyx RK45')
ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
ax2.plot3D(sol_cp[1], sol_cp[2], sol_cp[3], linewidth=0.5)

fig.suptitle('Lorenz attractor  (a={}, b={}, c={:.3f})'.format(a, b, c))
plt.tight_layout()

# --- Plot: x(t) time series ---
fig2, ax = plt.subplots(figsize=(10, 4))
ax.set_title('Lorenz — x(t) time series')
ax.set_xlabel('t')
ax.set_ylabel('x(t)')
ax.plot(sol_scipy.t, sol_scipy.y[0], 'b-',  lw=0.8, label='scipy RK45')
ax.plot(sol_cp[0],   sol_cp[1],      'r--', lw=0.8, label='compPhyx RK45')
ax.legend()
plt.tight_layout()

plt.show()
