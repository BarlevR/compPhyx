'''
compPhyx
Directory: examples/PDE/heatTransfer

Code: 2D heat equation via the Method of Lines.

      The PDE  ∂u/∂t = a * (∂²u/∂x² + ∂²u/∂y²)  is discretised in
      space on an (nx × ny) grid with finite differences, producing an
      nx*ny ODE system integrated forward in time.

      Boundary conditions (Dirichlet, fixed):
          u(x=0)  = bc_left    u(x=L)  = bc_right
          u(y=0)  = bc_bottom  u(y=H)  = bc_top

      Note: stability limit for explicit methods is roughly
            h ≤ dx² / (4*a)  (2D diffusion, factor of 4 not 2).
            With dx=dy=1, a=1 → h_max ≈ 0.25.
            scipy solve_ivp adapts its internal step automatically;
            compPhyx uses a fixed step set by t_eval, so t_eval must
            be dense enough to satisfy the stability limit.

Solvers compared:
    scipy.integrate.solve_ivp  (RK45)      — reference
    compPhyx.applications.HeatEquation2D  — METHOD / SPATIAL_SCHEME selectable

Author: Barlev Raymond
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import compPhyx.logo as logo
from compPhyx.applications import HeatEquation2D

print(logo.art)

# --- Pick compPhyx solver: 'Euler', 'RK2', 'RK3', 'RK4', 'RK45' ---
METHOD = 'RK45'

# --- Pick spatial scheme: 'CentralLaplacian' (2nd-order) or 'RichardsonLaplacian' (4th-order) ---
SPATIAL_SCHEME = 'CentralLaplacian'

# --- Physical parameters ---
thermal_diffusivity = 1.0   # thermal diffusivity (a)
dx                  = 1.0   # spatial step size in x
dy                  = 1.0   # spatial step size in y

# --- Spatial grid ---
nx = 50                     # grid points in x
ny = 50                     # grid points in y

# --- Boundary conditions (Dirichlet) ---
bc_left   = 0.0             # fixed temperature at x=0 edge
bc_right  = 0.0             # fixed temperature at x=L edge
bc_bottom = 0.0             # fixed temperature at y=0 edge
bc_top    = 10.0            # fixed temperature at y=H edge (hot top wall)

# --- Time span ---
tStart = 0.0
tEnd   = 500.0

# --- Problem setup ---
problem = HeatEquation2D(
    thermal_diffusivity=thermal_diffusivity,
    dx=dx,
    dy=dy,
    nx=nx,
    ny=ny,
    bc_left=bc_left,
    bc_right=bc_right,
    bc_bottom=bc_bottom,
    bc_top=bc_top,
    spatial_scheme=SPATIAL_SCHEME,
)

# h = tEnd / (len(t_eval)-1) must satisfy h ≤ dx²/(4*a) = 0.25
# Using h = 0.1 → 5001 points
t_eval = np.linspace(tStart, tEnd, 5001)

# --- scipy solve (reference) ---
sol_scipy = integrate.solve_ivp(problem.f, [tStart, tEnd], problem.r0,
                                method='RK45', t_eval=t_eval)

# --- compPhyx solve ---
sol_cp = problem.solve(method=METHOD, t_span=[tStart, tEnd], t_eval=t_eval)

# --- Helper: reshape flat solution at a given time index ---
def field(sol, idx):
    return sol.y[:, idx].reshape(nx, ny)

# --- Plot: temperature at grid centre vs time ---
cx, cy = nx // 2, ny // 2

fig, ax = plt.subplots(figsize=(8, 4))
fig.suptitle(f'2D Heat Equation — centre point temperature  '
             f'(a={thermal_diffusivity}, dx={dx}, dy={dy})')
ax.set_xlabel('Time t')
ax.set_ylabel(f'Temperature at ({cx}, {cy})')
ax.plot(sol_scipy.t, sol_scipy.y[cx*ny + cy], 'b-',  lw=0.8, label='scipy RK45')
ax.plot(sol_cp.t,    sol_cp.y[cx*ny + cy],    'r--', lw=1.2, label=f'compPhyx {METHOD}+{SPATIAL_SCHEME}')
ax.legend()
plt.tight_layout()

# --- Plot: temperature field snapshots ---
snap_indices = [0, len(t_eval)//4, len(t_eval)//2, len(t_eval)-1]
snap_labels  = ['t = 0', f't = {tEnd/4:.0f}', f't = {tEnd/2:.0f}', f't = {tEnd:.0f}']

fig2, axes2 = plt.subplots(2, 4, figsize=(16, 7))
fig2.suptitle(f'2D Heat Equation — temperature field snapshots  '
              f'(spatial: {SPATIAL_SCHEME})')

for col, (idx, label) in enumerate(zip(snap_indices, snap_labels)):
    u_scipy = field(sol_scipy, idx)
    u_cp    = field(sol_cp,    idx)
    vmin = min(u_scipy.min(), u_cp.min())
    vmax = max(u_scipy.max(), u_cp.max())

    im0 = axes2[0, col].imshow(u_scipy.T, origin='lower', aspect='auto',
                                vmin=vmin, vmax=vmax, cmap='hot')
    axes2[0, col].set_title(f'scipy RK45\n{label}')
    axes2[0, col].set_xlabel('x'); axes2[0, col].set_ylabel('y')
    plt.colorbar(im0, ax=axes2[0, col])

    im1 = axes2[1, col].imshow(u_cp.T, origin='lower', aspect='auto',
                                vmin=vmin, vmax=vmax, cmap='hot')
    axes2[1, col].set_title(f'compPhyx {METHOD}\n{label}')
    axes2[1, col].set_xlabel('x'); axes2[1, col].set_ylabel('y')
    plt.colorbar(im1, ax=axes2[1, col])

plt.tight_layout()
plt.show()
