'''
compPhyx
Directory: examples/ODE/rollingBall

Code: Ball in a parabolic bowl with initial velocity and periodic forcing.
      Damped driven harmonic oscillator in 2D.

      Potential:  U(x, y) = U0 * (x^2 + y^2)
      Forcing:    F(t) = A0 * sin(2*pi*t / tOsc),  direction angle phi
      Equations of motion:
          m*x'' = -xi*x' - 2*U0*x + F(t)*cos(phi)
          m*y'' = -xi*y' - 2*U0*y + F(t)*sin(phi)

      Reduced to 4 first-order ODEs via state vector r = [x, y, vx, vy].

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

# --- Parameters ---
m    = 1.0
U0   = 1.0
xi   = 0.1
A0   = 4.0
tOsc = 50.0
phi  = 45.0 / 180.0 * np.pi

# --- ODE right-hand side (with forcing) ---
def f_ODE(t, r):
    x, y   = r[0], r[1]
    vx, vy = r[2], r[3]
    F = A0 * np.sin(2*np.pi*t / tOsc)
    return np.array([
        vx,
        vy,
        -xi/m*vx - 2*U0/m*x + F*np.cos(phi),
        -xi/m*vy - 2*U0/m*y + F*np.sin(phi),
    ])

# --- Initial conditions ---
tStart = 0.0
tEnd   = 100.0
r0     = [2.0, 0.0, 0.0, 1.0]

# --- scipy solve ---
t_eval    = np.linspace(tStart, tEnd, 1001)
sol_scipy = integrate.solve_ivp(f_ODE, [tStart, tEnd], r0, method='RK45', t_eval=t_eval)

# --- compPhyx solve ---
nmax   = 1000
h      = (tEnd - tStart) / nmax
sol_cp = RK45(f_ODE, t0=tStart, y0=r0, nmax=nmax, h=h).solve()
# sol_cp[0] = t, sol_cp[1] = x, sol_cp[2] = y, sol_cp[3] = vx, sol_cp[4] = vy

# --- Plot: time series ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Forced ball in a bowl')

axes[0].set_xlabel('Time t')
axes[0].set_ylabel('x(t)')
axes[0].plot(sol_scipy.t,  sol_scipy.y[0], 'b-',  label='scipy RK45')
axes[0].plot(sol_cp[0],    sol_cp[1],       'r--', label='compPhyx RK45', linewidth=1.5)
axes[0].legend()

axes[1].set_xlabel('Time t')
axes[1].set_ylabel('y(t)')
axes[1].plot(sol_scipy.t,  sol_scipy.y[1], 'b-',  label='scipy RK45')
axes[1].plot(sol_cp[0],    sol_cp[2],       'r--', label='compPhyx RK45', linewidth=1.5)
axes[1].legend()

plt.tight_layout()

# --- Plot: xy trajectory ---
fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.set_aspect('equal')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Forced ball in a bowl — xy trajectory')
ax2.plot(sol_scipy.y[0], sol_scipy.y[1], 'b-',  label='scipy RK45')
ax2.plot(sol_cp[1],      sol_cp[2],       'r--', label='compPhyx RK45', linewidth=1.5)
ax2.legend()

plt.show()
