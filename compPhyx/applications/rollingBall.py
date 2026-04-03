'''
compPhyx.applications.rollingBall

Ball rolling in a parabolic bowl: damped 2D harmonic oscillator,
optionally driven by a periodic external force.

    Potential:  U(x, y) = U0 * (x^2 + y^2)
    Equations of motion:
        m*x'' = -xi*x' - 2*U0*x + F(t)*cos(phi)
        m*y'' = -xi*y' - 2*U0*y + F(t)*sin(phi)
    where F(t) = A0 * sin(2*pi*t / tOsc)

Reduced to a 4D first-order system via state vector r = [x, y, vx, vy].
Set A0=0 (default) for the unforced case.

Author: Barlev Raymond
'''

import numpy as np
from ._base import Application


class RollingBall(Application):
    '''
    Parameters
    ----------
    m       : float, mass
    U0      : float, bowl curvature (potential coefficient)
    xi      : float, damping coefficient
    A0      : float, forcing amplitude  (0 = no forcing)
    tOsc    : float, forcing period
    phi_deg : float, forcing direction in degrees
    r0      : array_like, initial state [x0, y0, vx0, vy0]
    '''

    def __init__(self, m=1.0, U0=1.0, xi=0.1,
                 A0=0.0, tOsc=50.0, phi_deg=0.0,
                 r0=None):
        self.m    = m
        self.U0   = U0
        self.xi   = xi
        self.A0   = A0
        self.tOsc = tOsc
        self.phi  = phi_deg / 180.0 * np.pi
        self.r0   = np.array(r0 if r0 is not None else [1.0, -0.5, 0.0, 0.0])

    def f(self, t, r):
        '''ODE right-hand side: f(t, r) → dr/dt'''
        x, y   = r[0], r[1]
        vx, vy = r[2], r[3]
        F = self.A0 * np.sin(2 * np.pi * t / self.tOsc)
        return np.array([
            vx,
            vy,
            -self.xi / self.m * vx - 2 * self.U0 / self.m * x + F * np.cos(self.phi),
            -self.xi / self.m * vy - 2 * self.U0 / self.m * y + F * np.sin(self.phi),
        ])
