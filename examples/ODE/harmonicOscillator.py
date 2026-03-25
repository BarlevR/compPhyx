'''
compPhyx
Directory: examples/ODE

Code: Harmonic oscillator (pendulum) solved with EulerSecondOrder.

Small-angle approximation:  theta'' = -(g/L) * theta
Analytical solution:        theta(t) = theta0 * cos(sqrt(g/L) * t)

Also demonstrates the real pendulum (nonlinear) and driven oscillator.

Author: Barlev Raymond
'''

import numpy as np
import matplotlib.pyplot as plt
import compPhyx.logo as logo
from compPhyx.timestepping import EulerSecondOrder

print(logo.art)

g      = 9.81
length = 2.0
c      = g / length

t0     = 0
nmax   = 200
h      = 0.1

# --- Small-angle pendulum ---
theta0  = 0.2
dtheta0 = 0.0

def small_angle(t, theta, dtheta):
    return -c * theta

sol = EulerSecondOrder(small_angle, t0, theta0, dtheta0, nmax, h).solve()

t_analytical   = np.linspace(0, nmax * h, 300)
theta_analytical = theta0 * np.cos(np.sqrt(c) * t_analytical)

plt.figure(1)
plt.title('Small-angle pendulum')
plt.xlabel('t (s)')
plt.ylabel('theta (rad)')
plt.plot(t_analytical, theta_analytical, 'r-', label='Analytical')
plt.scatter(sol[0], sol[1], s=5, label='Euler-Cromer')
plt.legend()

# --- Real pendulum (nonlinear, damped) ---
theta0 = 2.2
b      = 0.5

def real_pendulum(t, theta, dtheta):
    return -b * dtheta - c * np.sin(theta)

sol1 = EulerSecondOrder(real_pendulum, t0, theta0, dtheta0, nmax, h).solve()

plt.figure(2)
plt.title('Real pendulum (nonlinear, damped)')
plt.xlabel('t (s)')
plt.ylabel('theta (rad)')
plt.plot(sol1[0], sol1[1])

# --- Driven oscillator ---
nmax2 = 1000
b     = 0.1
d     = -1.0
omega = 1.0

def driven_oscillator(t, theta, dtheta):
    return -b * dtheta - c * np.sin(theta) - d * np.sin(omega * t)

sol2 = EulerSecondOrder(driven_oscillator, t0, theta0, dtheta0, nmax2, h).solve()

plt.figure(3)
plt.title('Driven oscillator')
plt.xlabel('t (s)')
plt.ylabel('theta (rad)')
plt.plot(sol2[0], sol2[1])

plt.show()
