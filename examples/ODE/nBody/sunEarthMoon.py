'''
compPhyx
Directory: examples/ODE/nBody

Code: Three-body problem — Sun, Earth, Moon

      Gravitational N-body system with 3 bodies using SI units.
      Bodies: Sun (0), Earth (1), Moon (2)

      Note: compPhyx uses a fixed time step; scipy RK45 is adaptive.
            Trajectory-level differences are expected for long integrations —
            both solutions are physically valid but will diverge over time.

Solvers compared:
    scipy.integrate.solve_ivp  (RK45)     — reference
    compPhyx.applications.NBodyGravity   — METHOD selectable below

Author: Barlev Raymond
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import compPhyx.logo as logo
from compPhyx.applications import NBodyGravity

print(logo.art)

# --- Pick compPhyx solver: 'Euler', 'RK2', 'RK3', 'RK4', 'RK45' ---
METHOD = 'RK45'

# --- Gravitational constant ---
G = 6.67430e-11  # m^3 / (kg * s^2)

# --- Masses ---
mass_sun   = 1.9884e+30  # kg
mass_earth = 5.9723e+24  # kg
mass_moon  = 7.3490e+22  # kg

# --- Distances ---
dist_sun_earth  = 1.4960e+11  # m  (average)
dist_earth_moon = 3.8500e+08  # m  (average)

# --- Orbital velocities ---
vel_earth = 29780.0  # m/s  (around Sun)
vel_moon  =  1022.0  # m/s  (around Earth)

# --- Time span: 10 years ---
year   = 60 * 60 * 24 * 365.25   # seconds
tStart = 0.0
tEnd   = 10.0 * year

# --- Initial conditions ---
#         x                    y                   z
pos_sun   = [0.0,              0.0,                0.0]
pos_earth = [dist_sun_earth,   0.0,                0.0]
pos_moon  = [dist_sun_earth,   dist_earth_moon,    0.0]

#              vx          vy          vz
vel_sun_vec   = [0.0,      0.0,        0.0]
vel_earth_vec = [0.0,      vel_earth,  0.0]
vel_moon_vec  = [-vel_moon, vel_earth, 0.0]

masses     = [mass_sun, mass_earth, mass_moon]
positions  = [pos_sun,  pos_earth,  pos_moon]
velocities = [vel_sun_vec, vel_earth_vec, vel_moon_vec]

# --- Problem setup ---
problem = NBodyGravity(
    masses=masses,
    positions=positions,
    velocities=velocities,
    G=G,
)

t_eval = np.linspace(tStart, tEnd, 100001)

# --- scipy solve (reference) ---
sol_scipy = integrate.solve_ivp(problem.f, [tStart, tEnd], problem.r0,
                                method='RK45', t_eval=t_eval,
                                rtol=1e-9, atol=1e-12)

# --- compPhyx solve ---
sol_cp = problem.solve(method=METHOD, t_span=[tStart, tEnd], t_eval=t_eval)

# --- Convenience: time in years ---
t_scipy = sol_scipy.t / year
t_cp    = sol_cp.t    / year

# --- Trajectory helpers ---
def pos_scipy(i): return problem.body_pos(sol_scipy, i)
def pos_cp(i):    return problem.body_pos(sol_cp,    i)

# --- Plot 1: all bodies ---
fig1, ax1 = plt.subplots(figsize=(7, 7))
fig1.suptitle('Sun, Earth, Moon — full trajectories')
ax1.set_aspect('equal')
ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)')
ax1.plot(*pos_scipy(0)[:2], 'r-',  lw=0.6, label='Sun  (scipy)')
ax1.plot(*pos_scipy(1)[:2], 'b-',  lw=0.6, label='Earth (scipy)')
ax1.plot(*pos_scipy(2)[:2], color='gray', lw=0.4, label='Moon (scipy)')
ax1.plot(*pos_cp(0)[:2],    'r--', lw=0.8, label=f'Sun  ({METHOD})')
ax1.plot(*pos_cp(1)[:2],    'b--', lw=0.8, label=f'Earth ({METHOD})')
ax1.plot(*pos_cp(2)[:2],    color='lightgray', ls='--', lw=0.6, label=f'Moon ({METHOD})')
ax1.legend(fontsize=8)
plt.tight_layout()

# --- Plot 2: Sun-Earth distance vs time ---
dist_se_scipy = np.linalg.norm(pos_scipy(0) - pos_scipy(1), axis=0)
dist_se_cp    = np.linalg.norm(pos_cp(0)    - pos_cp(1),    axis=0)

fig2, ax2 = plt.subplots(figsize=(9, 4))
fig2.suptitle('Sun–Earth distance')
ax2.set_xlabel('Time (years)'); ax2.set_ylabel('Distance (m)')
ax2.plot(t_scipy, dist_se_scipy, 'b-',  lw=0.8, label='scipy RK45')
ax2.plot(t_cp,    dist_se_cp,    'r--', lw=1.0, label=f'compPhyx {METHOD}')
ax2.legend()
plt.tight_layout()

# --- Plot 3: Moon trajectory relative to Earth ---
fig3, ax3 = plt.subplots(figsize=(6, 6))
fig3.suptitle('Moon orbit relative to Earth')
ax3.set_aspect('equal')
ax3.set_xlabel('x wrt Earth (m)'); ax3.set_ylabel('y wrt Earth (m)')
moon_rel_scipy = pos_scipy(2) - pos_scipy(1)
moon_rel_cp    = pos_cp(2)    - pos_cp(1)
ax3.plot(*moon_rel_scipy[:2], color='gray', lw=0.5, label='scipy RK45')
ax3.plot(*moon_rel_cp[:2],    color='lightgray', ls='--', lw=0.8, label=f'compPhyx {METHOD}')
ax3.legend()
plt.tight_layout()

# --- Plot 4: Earth-Moon distance vs time ---
dist_em_scipy = np.linalg.norm(pos_scipy(2) - pos_scipy(1), axis=0)
dist_em_cp    = np.linalg.norm(pos_cp(2)    - pos_cp(1),    axis=0)

fig4, ax4 = plt.subplots(figsize=(9, 4))
fig4.suptitle('Earth–Moon distance')
ax4.set_xlabel('Time (years)'); ax4.set_ylabel('Distance (m)')
ax4.plot(t_scipy, dist_em_scipy, color='gray', lw=0.8, label='scipy RK45')
ax4.plot(t_cp,    dist_em_cp,    color='lightgray', ls='--', lw=1.0, label=f'compPhyx {METHOD}')
ax4.legend()
plt.tight_layout()

# --- Plot 5: Moon orbit relative to Earth (exaggerated scale) ---
fig5, ax5 = plt.subplots(figsize=(7, 7))
fig5.suptitle('All bodies — Moon orbit exaggerated (100×)')
ax5.set_aspect('equal')
ax5.set_xlabel('x (m)'); ax5.set_ylabel('y (m)')
ax5.plot(*pos_scipy(0)[:2], 'r-', lw=0.6, label='Sun')
ax5.plot(*pos_scipy(1)[:2], 'b-', lw=0.6, label='Earth')
moon_exag = pos_scipy(1) + 100 * (pos_scipy(2) - pos_scipy(1))
ax5.plot(*moon_exag[:2], color='gray', lw=0.4, label='Moon (100× exaggerated)')
ax5.legend(fontsize=8)
plt.tight_layout()

plt.show()
