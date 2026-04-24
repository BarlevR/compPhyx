'''
compPhyx
Directory: examples/ODE/nBody

Code: Four-body problem — Sun, Earth, Moon, Spaceship
      Scenario: spaceship escapes Earth gravity (v = 1.5 × circular)

      At 1.5× the circular orbit velocity the spaceship has enough
      energy to escape Earth's gravitational well and begin orbiting
      the Sun independently.

      Bodies: Sun (0), Earth (1), Moon (2), Spaceship (3)

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
G = 6.67430e-11

# --- Masses ---
mass_sun   = 1.9884e+30
mass_earth = 5.9723e+24
mass_moon  = 7.3490e+22

# --- Distances ---
dist_sun_earth  = 1.4960e+11
dist_earth_moon = 3.8500e+08
dist_orbit      = 4.2164e+07

# --- Orbital velocities ---
vel_earth = 29780.0
vel_moon  =  1022.0
vel_orbit =  3075.0

velocity_factor = 1.5   # escape trajectory

# --- Time span: 2 years ---
# The ship escapes Earth within ~25 days and reaches 0.75 AU within 1 year.
# 16 years is far too long for a fixed-step solver — errors accumulate to ~AU scale.
# 2 years is enough to show the full escape and heliocentric orbit.
year   = 60 * 60 * 24 * 365.25
tStart = 0.0
tEnd   = 8.0 * year

# --- Initial conditions ---
pos_sun      = [0.0,             0.0,             0.0]
pos_earth    = [dist_sun_earth,  0.0,             0.0]
pos_moon     = [dist_sun_earth,  dist_earth_moon, 0.0]
pos_ship     = [dist_sun_earth,  dist_orbit,      0.0]

vel_sun_vec   = [0.0,                          0.0,       0.0]
vel_earth_vec = [0.0,                          vel_earth, 0.0]
vel_moon_vec  = [-vel_moon,                    vel_earth, 0.0]
vel_ship_vec  = [-velocity_factor * vel_orbit, vel_earth, 0.0]

masses     = [mass_sun, mass_earth, mass_moon, 0.0]
positions  = [pos_sun,  pos_earth,  pos_moon,  pos_ship]
velocities = [vel_sun_vec, vel_earth_vec, vel_moon_vec, vel_ship_vec]

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

t_years_scipy = sol_scipy.t / year
t_years_cp    = sol_cp.t    / year

def pos_scipy(i): return problem.body_pos(sol_scipy, i)
def pos_cp(i):    return problem.body_pos(sol_cp,    i)

# --- Plot 1: all bodies ---
fig1, ax1 = plt.subplots(figsize=(7, 7))
fig1.suptitle(f'Earth escape (v = {velocity_factor}× circular) — full view')
ax1.set_aspect('equal')
ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)')
ax1.plot(*pos_scipy(0)[:2], 'r-',     lw=0.6, label='Sun')
ax1.plot(*pos_scipy(1)[:2], 'b-',     lw=0.6, label='Earth')
ax1.plot(*pos_scipy(2)[:2], color='gray',   lw=0.4, label='Moon')
ax1.plot(*pos_scipy(3)[:2], color='purple', lw=0.5, label='Spaceship (scipy)')
ax1.plot(*pos_cp(3)[:2],    color='violet', ls='--', lw=0.6, label=f'Spaceship ({METHOD})')
ax1.legend(fontsize=8)
plt.tight_layout()

# --- Plot 2: spaceship and Moon relative to Earth ---
fig2, ax2 = plt.subplots(figsize=(7, 7))
fig2.suptitle('Spaceship and Moon — relative to Earth')
ax2.set_aspect('equal')
ax2.set_xlabel('x wrt Earth (m)'); ax2.set_ylabel('y wrt Earth (m)')
ax2.plot(*(pos_scipy(2) - pos_scipy(1))[:2], color='gray',   lw=0.5, label='Moon (scipy)')
ax2.plot(*(pos_scipy(3) - pos_scipy(1))[:2], color='purple', lw=0.5, label='Spaceship (scipy)')
ax2.plot(*(pos_cp(3)    - pos_cp(1))[:2],    color='violet', ls='--', lw=0.8, label=f'Spaceship ({METHOD})')
ax2.legend()
plt.tight_layout()

# --- Plot 3: zoomed view around Earth ---
zoom = 3 * dist_earth_moon
fig3, ax3 = plt.subplots(figsize=(7, 7))
fig3.suptitle('Spaceship escape — zoomed around Earth')
ax3.set_aspect('equal')
ax3.set_xlim([-zoom, zoom]); ax3.set_ylim([-zoom, zoom])
ax3.set_xlabel('x wrt Earth (m)'); ax3.set_ylabel('y wrt Earth (m)')
ax3.plot(*(pos_scipy(2) - pos_scipy(1))[:2], color='gray',   lw=0.5, label='Moon')
ax3.plot(*(pos_scipy(3) - pos_scipy(1))[:2], color='purple', lw=0.5, label='Spaceship (scipy)')
ax3.plot(*(pos_cp(3)    - pos_cp(1))[:2],    color='violet', ls='--', lw=0.8, label=f'Spaceship ({METHOD})')
ax3.legend()
plt.tight_layout()

# --- Plot 4: spaceship-Earth distance vs time ---
dist_ship_earth_scipy = np.linalg.norm(pos_scipy(3) - pos_scipy(1), axis=0)
dist_ship_earth_cp    = np.linalg.norm(pos_cp(3)    - pos_cp(1),    axis=0)

fig4, ax4 = plt.subplots(figsize=(9, 4))
fig4.suptitle('Spaceship–Earth distance')
ax4.set_xlabel('Time (years)'); ax4.set_ylabel('Distance (m)')
ax4.plot(t_years_scipy, dist_ship_earth_scipy, color='purple', lw=0.8, label='scipy RK45')
ax4.plot(t_years_cp,    dist_ship_earth_cp,    color='violet', ls='--', lw=1.0, label=f'compPhyx {METHOD}')
ax4.legend()
plt.tight_layout()

plt.show()
