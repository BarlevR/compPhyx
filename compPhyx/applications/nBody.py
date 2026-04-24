'''
compPhyx.applications.nBody

N-body gravitational system.

Each body exerts a gravitational force on every other body:

    a_i = Σ_{j≠i}  G * m_j / |r_j - r_i|³  *  (r_j - r_i)

State vector layout (flat, 6N elements for 3D):
    [ r1x r1y r1z  r2x r2y r2z  ...  v1x v1y v1z  v2x v2y v2z  ... ]
     |------ N positions (3N) ------|----- N velocities (3N) ------|

An optional external_force(t, pos, vel) callable can be supplied to add
time-dependent forces (e.g. engine thrust) to any body.  It receives
positions and velocities as (N, 3) arrays and must return an (N, 3)
array of additional accelerations.

Author: Barlev Raymond
'''

import numpy as np
from ._base import Application


class NBodyGravity(Application):
    '''
    Parameters
    ----------
    masses         : array_like, shape (N,), masses of each body in kg
    positions      : array_like, shape (N, 3), initial positions in m
    velocities     : array_like, shape (N, 3), initial velocities in m/s
    G              : float, gravitational constant (default SI value)
    external_force : callable or None
                     external_force(t, pos, vel) -> (N, 3) ndarray
                     Additional acceleration applied to each body.
                     pos and vel are (N, 3) arrays at the current step.
    '''

    def __init__(self, masses, positions, velocities,
                 G=6.674e-11, external_force=None):
        self.masses         = np.asarray(masses,    dtype=float)
        self.N              = len(self.masses)
        self.G              = G
        self.external_force = external_force

        pos = np.asarray(positions,  dtype=float)   # (N, 3)
        vel = np.asarray(velocities, dtype=float)   # (N, 3)
        self.r0 = np.concatenate([pos.ravel(), vel.ravel()])

    # ------------------------------------------------------------------
    def f(self, t, state):
        '''Gravitational N-body ODE right-hand side.'''
        pos = state[:3*self.N].reshape(self.N, 3)
        vel = state[3*self.N:].reshape(self.N, 3)

        acc = np.zeros((self.N, 3))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    r_ij = pos[j] - pos[i]
                    dist = np.linalg.norm(r_ij)
                    acc[i] += self.G * self.masses[j] / dist**3 * r_ij

        if self.external_force is not None:
            acc += self.external_force(t, pos, vel)

        return np.concatenate([vel.ravel(), acc.ravel()])

    # ------------------------------------------------------------------
    # Convenience helpers for extracting body trajectories from a solution
    # ------------------------------------------------------------------
    def body_pos(self, sol, i):
        '''
        Position trajectory of body i.

        Returns
        -------
        ndarray, shape (3, n_times)  — rows are x, y, z
        '''
        return sol.y[3*i : 3*(i+1)]

    def body_vel(self, sol, i):
        '''
        Velocity trajectory of body i.

        Returns
        -------
        ndarray, shape (3, n_times)  — rows are vx, vy, vz
        '''
        return sol.y[3*self.N + 3*i : 3*self.N + 3*(i+1)]
