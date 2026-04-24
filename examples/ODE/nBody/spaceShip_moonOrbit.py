'''
compPhyx
Directory: examples/ODE/nBody

Code: Four-body problem — Sun, Earth, Moon, Spaceship
      Scenario: engine burn to brake into Moon orbit

      The spaceship starts on a Moon-encounter trajectory (1.34× circular).
      When it approaches the Moon, the engine fires for a short window
      (0.11 to 0.12 years) applying a braking force opposing the
      spaceship's velocity relative to the Moon.  This reduces the
      relative speed and allows the Moon's gravity to capture the ship.

      The thrust is supplied as an external_force callable — it receives
      positions and velocities as (N, 3) arrays and returns (N, 3)
      accelerations.

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

velocity_factor = 1.34   # same as moon encounter

# --- Engine parameters ---
year           = 60 * 60 * 24 * 365.25
thrust_magnitude = 27.0e+06   # m/s² (acceleration, mass ~1 kg for simplicity)
thrust_t_on    = 0.11 * year  # engine start time
thrust_t_off   = 0.12 * year  # engine stop time

# --- External force: braking thrust opposing velocity relative to Moon ---
def moon_braking_thrust(t, pos, vel):
    '''
    Fires the spaceship engine (body 3) during the burn window.
    Thrust direction opposes the spaceship's velocity relative to the Moon.
    '''
    force = np.zeros_like(pos)
    burn  = thrust_magnitude * (
        np.heaviside(t - thrust_t_on,  1.0) -
        np.heaviside(t - thrust_t_off, 1.0)
    )
    if burn > 0:
        rel_vel = vel[3] - vel[2]           # spaceship vel minus Moon vel
        norm    = np.linalg.norm(rel_vel)
        if norm > 0:
            force[3] = -burn * rel_vel / norm
    return force

# --- Time span: 0.3 years ---
tStart = 0.0
tEnd   = 0.3 * year

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
    external_force=moon_braking_thrust,
)

# 1 000 001 points → h ≈ 9.5 s, needed to resolve the close Moon flyby
t_eval = np.linspace(tStart, tEnd, 1000001)

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

# --- Plot 1: spaceship and Moon relative to Earth ---
zoom = 2 * dist_earth_moon
fig1, ax1 = plt.subplots(figsize=(7, 7))
fig1.suptitle('Moon orbit insertion — relative to Earth')
ax1.set_aspect('equal')
ax1.set_xlim([-zoom, zoom]); ax1.set_ylim([-zoom, zoom])
ax1.set_xlabel('x wrt Earth (m)'); ax1.set_ylabel('y wrt Earth (m)')
ax1.plot(*(pos_scipy(2) - pos_scipy(1))[:2], color='gray',   lw=0.6, label='Moon (scipy)')
ax1.plot(*(pos_scipy(3) - pos_scipy(1))[:2], color='purple', lw=0.6, label='Spaceship (scipy)')
ax1.plot(*(pos_cp(3)    - pos_cp(1))[:2],    color='violet', ls='--', lw=0.8, label=f'Spaceship ({METHOD})')
ax1.axvline(0, color='k', lw=0.3); ax1.axhline(0, color='k', lw=0.3)
ax1.legend()
plt.tight_layout()

# --- Plot 2: spaceship-Earth distance vs time ---
dist_ship_earth_scipy = np.linalg.norm(pos_scipy(3) - pos_scipy(1), axis=0)
dist_ship_earth_cp    = np.linalg.norm(pos_cp(3)    - pos_cp(1),    axis=0)

fig2, ax2 = plt.subplots(figsize=(9, 4))
fig2.suptitle('Spaceship–Earth distance')
ax2.set_xlabel('Time (years)'); ax2.set_ylabel('Distance (m)')
ax2.axvspan(thrust_t_on/year, thrust_t_off/year, alpha=0.15, color='orange', label='Engine burn')
ax2.plot(t_years_scipy, dist_ship_earth_scipy, color='purple', lw=0.8, label='scipy RK45')
ax2.plot(t_years_cp,    dist_ship_earth_cp,    color='violet', ls='--', lw=1.0, label=f'compPhyx {METHOD}')
ax2.legend()
plt.tight_layout()

# --- Plot 3: spaceship-Moon distance vs time ---
dist_ship_moon_scipy = np.linalg.norm(pos_scipy(3) - pos_scipy(2), axis=0)
dist_ship_moon_cp    = np.linalg.norm(pos_cp(3)    - pos_cp(2),    axis=0)

fig3, ax3 = plt.subplots(figsize=(9, 4))
fig3.suptitle('Spaceship–Moon distance')
ax3.set_xlabel('Time (years)'); ax3.set_ylabel('Distance (m)')
ax3.axvspan(thrust_t_on/year, thrust_t_off/year, alpha=0.15, color='orange', label='Engine burn')
ax3.plot(t_years_scipy, dist_ship_moon_scipy, color='purple', lw=0.8, label='scipy RK45')
ax3.plot(t_years_cp,    dist_ship_moon_cp,    color='violet', ls='--', lw=1.0, label=f'compPhyx {METHOD}')
ax3.legend()
plt.tight_layout()

# --- Plot 4: spaceship trajectory relative to Moon (zoomed) ---
zoom_moon = 0.2 * dist_earth_moon
fig4, ax4 = plt.subplots(figsize=(6, 6))
fig4.suptitle('Spaceship trajectory — relative to Moon')
ax4.set_aspect('equal')
ax4.set_xlim([-zoom_moon, zoom_moon]); ax4.set_ylim([-zoom_moon, zoom_moon])
ax4.set_xlabel('x wrt Moon (m)'); ax4.set_ylabel('y wrt Moon (m)')
ax4.plot(*(pos_scipy(3) - pos_scipy(2))[:2], color='purple', lw=0.6, label='scipy RK45')
ax4.plot(*(pos_cp(3)    - pos_cp(2))[:2],    color='violet', ls='--', lw=0.8, label=f'compPhyx {METHOD}')
ax4.plot(0, 0, 'o', color='gray', ms=8, label='Moon')
ax4.legend()
plt.tight_layout()

plt.show()
