'''
compPhyx
Subpackage: ode
Module: euler

Euler ODE solvers as classes.

  Euler             — first-order ODE:  dy/dt = f(t, y)
  EulerSecondOrder  — second-order ODE: y'' = f(t, y, y')
                      Uses the Euler-Cromer (symplectic) update so that
                      oscillatory solutions conserve energy.
'''

import numpy as np


class Euler:
    '''
    Euler method for a first-order ODE: dy/dt = f(t, y)

    Parameters
    ----------
    f     : callable, f(t, y) — the ODE right-hand side
    t0    : float, initial time
    y0    : float, initial value of y
    nmax  : int, number of steps
    h     : float, step size
    '''

    def __init__(self, f, t0, y0, nmax, h):
        self.f    = f
        self.t0   = float(t0)
        self.y0   = float(y0)
        self.nmax = int(nmax)
        self.h    = float(h)

    def solve(self):
        '''
        Returns
        -------
        np.ndarray, shape (2, nmax+1) — rows are [t_values, y_values]
        '''
        t = self.t0
        y = self.y0
        t_values = [t]
        y_values = [y]
        for _ in range(self.nmax):
            y = y + self.f(t, y) * self.h
            t = t + self.h
            t_values.append(t)
            y_values.append(y)
        return np.array([t_values, y_values])


class EulerSecondOrder:
    '''
    Euler-Cromer (symplectic Euler) method for a second-order ODE:
        y'  = dy
        dy' = f(t, y, dy)

    Position is updated first, then velocity uses the new position.
    This preserves energy in oscillatory systems unlike standard Euler.

    Parameters
    ----------
    f     : callable, f(t, y, dy) — the second derivative right-hand side
    t0    : float, initial time
    y0    : float, initial value of y
    dy0   : float, initial value of dy/dt
    nmax  : int, number of steps
    h     : float, step size
    '''

    def __init__(self, f, t0, y0, dy0, nmax, h):
        self.f    = f
        self.t0   = float(t0)
        self.y0   = float(y0)
        self.dy0  = float(dy0)
        self.nmax = int(nmax)
        self.h    = float(h)

    def solve(self):
        '''
        Returns
        -------
        np.ndarray, shape (3, nmax+1) — rows are [t_values, y_values, dy_values]
        '''
        t  = self.t0
        y  = self.y0
        dy = self.dy0
        t_values  = [t]
        y_values  = [y]
        dy_values = [dy]
        for _ in range(self.nmax):
            y  = y + dy * self.h                      # position first (Euler-Cromer)
            dy = dy + self.f(t, y, dy) * self.h       # velocity uses new position
            t  = t + self.h
            t_values.append(t)
            y_values.append(y)
            dy_values.append(dy)
        return np.array([t_values, y_values, dy_values])
