'''
compPhyx.applications.lorenz

The Lorenz system — a chaotic strange attractor.

    x' = a*(y - x)
    y' = x*(b - z) - y
    z' = x*y - c*z

For b > 24.74 the system is chaotic and trajectories converge to
the Lorenz butterfly (strange attractor).
'''

import numpy as np


class LorenzSystem:
    '''
    Parameters
    ----------
    a  : float, Prandtl number (default 10)
    b  : float, Rayleigh number (default 50, > 24.74 for chaos)
    c  : float, geometric factor (default 8/3)
    r0 : array_like, initial state [x0, y0, z0]
    '''

    def __init__(self, a=10.0, b=50.0, c=8.0/3.0, r0=None):
        self.a  = a
        self.b  = b
        self.c  = c
        self.r0 = np.array(r0 if r0 is not None else [1.0, 0.0, 0.0])

    def f(self, t, r):
        '''ODE right-hand side: f(t, r) → dr/dt'''
        x, y, z = r
        return np.array([
            self.a * (y - x),
            x * (self.b - z) - y,
            x * y - self.c * z,
        ])
