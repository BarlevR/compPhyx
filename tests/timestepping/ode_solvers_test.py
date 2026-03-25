'''
compPhyx
Directory: tests/timestepping

Code: Compare ODE solvers against an analytical solution.

Problem: radioactive decay  dy/dt = -y,  y(0) = 1
Analytical solution:        y(t) = e^{-t}

Runs Euler, RK4, and RK45 with the same step size and prints
the absolute error at the final timestep for each method.

Author: Barlev Raymond
'''

import numpy as np
import compPhyx.logo as logo
from compPhyx.timestepping import Euler, RK4, RK45

def f(t, y):
    return -y

t0   = 0
y0   = 1.0
h    = 0.1
nmax = 100
t_end = t0 + nmax * h

analytical = np.exp(-t_end)

sol_euler = Euler(f, t0, y0, nmax, h).solve()
sol_rk4   = RK4(f, t0, y0, nmax, h).solve()
sol_rk45  = RK45(f, t0, y0, nmax, h).solve()

err_euler = abs(sol_euler[1, -1] - analytical)
err_rk4   = abs(sol_rk4[1, -1]  - analytical)
err_rk45  = abs(sol_rk45[1, -1] - analytical)

print(logo.art)
print(f"ODE solver comparison — radioactive decay (dy/dt = -y)")
print(f"Step size h = {h},  t in [0, {t_end}],  analytical y({t_end}) = {analytical:.10f}\n")
print(f"  {'Method':<10}  {'y(t_end)':<18}  {'Abs error'}")
print(f"  {'-'*10}  {'-'*18}  {'-'*18}")
print(f"  {'Euler':<10}  {sol_euler[1,-1]:<18.10f}  {err_euler:.6e}")
print(f"  {'RK4':<10}  {sol_rk4[1,-1]:<18.10f}  {err_rk4:.6e}")
print(f"  {'RK45':<10}  {sol_rk45[1,-1]:<18.10f}  {err_rk45:.6e}")
