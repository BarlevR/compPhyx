'''
compPhyx
Directory: examples/ODE

Code: Compare our ODE solvers against scipy's solve_ivp methods
      on the driven oscillator problem.

Author: Barlev Raymond
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import compPhyx.logo as logo
from compPhyx.timestepping import EulerSecondOrder

print(logo.art)

g     = 9.81
L     = 2.0
c     = g / L
b     = 0.1
d     = -1.0
omega = 1.0

t0      = 0
t_end   = 100
theta0  = 2.0
dtheta0 = 0.0
nmax    = 200
h       = 0.5
t_eval  = np.linspace(t0, t_end, 201)

# --- Our Euler-Cromer solver ---
def driven_euler(t, theta, dtheta):
    return -b * dtheta - c * np.sin(theta) - d * np.sin(omega * t)

sol_euler = EulerSecondOrder(driven_euler, t0, theta0, dtheta0, nmax, h).solve()

# --- scipy solve_ivp (system form) ---
def driven_system(t, state):
    theta, dtheta = state
    return [dtheta, -b * dtheta - c * np.sin(theta) - d * np.sin(omega * t)]

ic     = [theta0, dtheta0]
t_span = [t0, t_end]

sol_RK45   = integrate.solve_ivp(driven_system, t_span, ic, method='RK45',   t_eval=t_eval)
sol_RK23   = integrate.solve_ivp(driven_system, t_span, ic, method='RK23',   t_eval=t_eval)
sol_DOP853 = integrate.solve_ivp(driven_system, t_span, ic, method='DOP853', t_eval=t_eval)
sol_Radau  = integrate.solve_ivp(driven_system, t_span, ic, method='Radau',  t_eval=t_eval)
sol_BDF    = integrate.solve_ivp(driven_system, t_span, ic, method='BDF',    t_eval=t_eval)
sol_LSODA  = integrate.solve_ivp(driven_system, t_span, ic, method='LSODA',  t_eval=t_eval)

# --- Plot solutions ---
plt.figure(1)
plt.title('Driven oscillator — all methods')
plt.xlabel('t')
plt.ylabel('theta')
plt.plot(sol_euler[0],  sol_euler[1],    label='Euler-Cromer (ours)')
plt.plot(sol_RK45.t,    sol_RK45.y[0],   label='RK45')
plt.plot(sol_RK23.t,    sol_RK23.y[0],   label='RK23')
plt.plot(sol_DOP853.t,  sol_DOP853.y[0], label='DOP853')
plt.plot(sol_Radau.t,   sol_Radau.y[0],  label='Radau')
plt.plot(sol_BDF.t,     sol_BDF.y[0],    label='BDF')
plt.plot(sol_LSODA.t,   sol_LSODA.y[0],  label='LSODA')
plt.legend()

# --- Plot deviation from RK45 (reference) ---
plt.figure(2)
plt.title('Deviation from RK45 (reference)')
plt.xlabel('t')
plt.ylabel('theta - theta_RK45')
plt.plot(sol_RK23.t,   sol_RK45.y[0] - sol_RK23.y[0],   label='RK23',   color='blue')
plt.plot(sol_DOP853.t, sol_RK45.y[0] - sol_DOP853.y[0], label='DOP853', color='red')
plt.plot(sol_Radau.t,  sol_RK45.y[0] - sol_Radau.y[0],  label='Radau',  color='green')
plt.plot(sol_BDF.t,    sol_RK45.y[0] - sol_BDF.y[0],    label='BDF',    color='orange')
plt.plot(sol_LSODA.t,  sol_RK45.y[0] - sol_LSODA.y[0],  label='LSODA',  color='purple')
plt.legend()

plt.show()
