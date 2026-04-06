'''
compPhyx
Directory: examples/ODE/heatTransfer

Code: 1D heat equation via the Method of Lines.

      The PDE  ∂u/∂t = a * ∂²u/∂x²  is discretised in space with
      finite differences, producing a 100D ODE system integrated
      forward in time.

      Boundary conditions (Dirichlet, fixed):
          u(x=0) = bc_left
          u(x=L) = bc_right

      Note: Euler and RK2 may become unstable near the stability limit
            h ≈ dx² / (2*a). RK3, RK4, RK45 are stable for this step size.

Solvers compared:
    scipy.integrate.solve_ivp  (RK45)       — reference
    compPhyx.applications.HeatEquation1D   — METHOD selectable below

Author: Barlev Raymond
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import compPhyx.logo as logo
from compPhyx.applications import HeatEquation1D

print(logo.art)

# --- Pick compPhyx solver: 'Euler', 'RK2', 'RK3', 'RK4', 'RK45' ---
METHOD = 'RK45'

# --- Physical parameters ---
thermal_diffusivity = 1.0   # thermal diffusivity (a)
dx                  = 1.0   # spatial step size

# --- Spatial grid ---
n_points = 100              # number of grid points

# --- Boundary conditions (Dirichlet) ---
bc_left  = 1.0              # fixed temperature at left end
bc_right = 10.0             # fixed temperature at right end

# --- Time span ---
tStart = 0.0
tEnd   = 5000.0

# --- Problem setup ---
problem = HeatEquation1D(
    thermal_diffusivity=thermal_diffusivity,
    dx=dx,
    n_points=n_points,
    bc_left=bc_left,
    bc_right=bc_right,
)

t_eval = np.linspace(tStart, tEnd, 10001)

# --- scipy solve (reference) ---
sol_scipy = integrate.solve_ivp(problem.f, [tStart, tEnd], problem.r0,
                                method='RK45', t_eval=t_eval)

# --- compPhyx solve ---
sol_cp = problem.solve(method=METHOD, t_span=[tStart, tEnd], t_eval=t_eval)

# --- Plot: temperature at midpoint vs time ---
mid = n_points // 2

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'1D Heat Equation  (a={thermal_diffusivity}, dx={dx}, '
             f'BC: left={bc_left}, right={bc_right})')

axes[0].set_xlabel('Time t')
axes[0].set_ylabel(f'Temperature at grid point {mid}')
axes[0].plot(sol_scipy.t, sol_scipy.y[mid], 'b-',  lw=0.8, label='scipy RK45')
axes[0].plot(sol_cp.t,    sol_cp.y[mid],    'r--', lw=1.2, label=f'compPhyx {METHOD}')
axes[0].legend()

# --- Plot: steady-state temperature profile ---
axes[1].set_xlabel('Grid point')
axes[1].set_ylabel('Temperature')
axes[1].set_title('Temperature profile at t = tEnd')
axes[1].plot(sol_scipy.y[:, -1], 'b-',  label='scipy RK45')
axes[1].plot(sol_cp.y[:, -1],    'r--', label=f'compPhyx {METHOD}')
axes[1].legend()

plt.tight_layout()

# --- Plot: contour — scipy reference ---
x_grid, t_grid = np.meshgrid(np.arange(n_points), sol_scipy.t)

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('1D Heat Equation — temperature field u(x, t)')

cf1 = axes2[0].contourf(x_grid, t_grid, sol_scipy.y.T, levels=50, cmap='hot')
axes2[0].set_xlabel('Grid point'); axes2[0].set_ylabel('Time t')
axes2[0].set_title('scipy RK45')
plt.colorbar(cf1, ax=axes2[0], label='Temperature')

cf2 = axes2[1].contourf(x_grid, t_grid, sol_cp.y.T, levels=50, cmap='hot')
axes2[1].set_xlabel('Grid point'); axes2[1].set_ylabel('Time t')
axes2[1].set_title(f'compPhyx {METHOD}')
plt.colorbar(cf2, ax=axes2[1], label='Temperature')

plt.tight_layout()
plt.show()
