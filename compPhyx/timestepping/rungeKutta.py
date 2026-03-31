'''
compPhyx
Subpackage: ode
Module: rungeKutta

Runge-Kutta ODE solvers as classes.

  RK4   — classical 4th-order Runge-Kutta
  RK45  — Runge-Kutta-Fehlberg 5th-order method
'''

import numpy as np


class RK4:
    '''
    Classical 4th-order Runge-Kutta method for dy/dt = f(t, y)

    Parameters
    ----------
    f     : callable, f(t, y) — the ODE right-hand side
    t0    : float, initial time
    y0    : scalar or array_like, initial value of y
    nmax  : int, number of steps
    h     : float, step size
    '''

    def __init__(self, f, t0, y0, nmax, h):
        self.f    = f
        self.t0   = float(t0)
        self.y0   = np.asarray(y0, dtype=float)
        self.nmax = int(nmax)
        self.h    = float(h)

    def solve(self):
        '''
        Returns
        -------
        np.ndarray, shape (d+1, nmax+1) — row 0 is t, rows 1..d are state components.
        For scalar y0, d=1 and shape is (2, nmax+1), matching the previous interface.
        '''
        t = self.t0
        y = self.y0
        t_values = [t]
        y_values = [y]
        for _ in range(self.nmax):
            k1 = self.h * self.f(t, y)
            k2 = self.h * self.f(t + self.h/2, y + k1/2)
            k3 = self.h * self.f(t + self.h/2, y + k2/2)
            k4 = self.h * self.f(t + self.h,   y + k3)
            y  = y + (k1 + 2*k2 + 2*k3 + k4) / 6
            t  = t + self.h
            t_values.append(t)
            y_values.append(y)
        t_arr = np.array(t_values)
        y_arr = np.array(y_values)
        if y_arr.ndim == 1:
            y_arr = y_arr[np.newaxis]   # (1, nmax+1)
        else:
            y_arr = y_arr.T             # (d, nmax+1)
        return np.vstack([t_arr[np.newaxis], y_arr])


class RK45:
    '''
    Runge-Kutta-Fehlberg (RK45) method for dy/dt = f(t, y).
    Uses the 5th-order (Fehlberg) update coefficients.

    Parameters
    ----------
    f     : callable, f(t, y) — the ODE right-hand side
    t0    : float, initial time
    y0    : scalar or array_like, initial value of y
    nmax  : int, number of steps
    h     : float, step size
    '''

    def __init__(self, f, t0, y0, nmax, h):
        self.f    = f
        self.t0   = float(t0)
        self.y0   = np.asarray(y0, dtype=float)
        self.nmax = int(nmax)
        self.h    = float(h)

    def solve(self):
        '''
        Returns
        -------
        np.ndarray, shape (d+1, nmax+1) — row 0 is t, rows 1..d are state components.
        For scalar y0, d=1 and shape is (2, nmax+1), matching the previous interface.
        '''
        t = self.t0
        y = self.y0
        t_values = [t]
        y_values = [y]
        for _ in range(self.nmax):
            k1 = self.h * self.f(t, y)
            k2 = self.h * self.f(t + self.h/4,      y + k1/4)
            k3 = self.h * self.f(t + 3*self.h/8,    y + 3*k1/32      + 9*k2/32)
            k4 = self.h * self.f(t + 12*self.h/13,  y + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
            k5 = self.h * self.f(t + self.h,         y + 439*k1/216   - 8*k2         + 3680*k3/513  - 845*k4/4104)
            k6 = self.h * self.f(t + self.h/2,       y - 8*k1/27      + 2*k2         - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)
            y  = y + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55
            t  = t + self.h
            t_values.append(t)
            y_values.append(y)
        t_arr = np.array(t_values)
        y_arr = np.array(y_values)
        if y_arr.ndim == 1:
            y_arr = y_arr[np.newaxis]   # (1, nmax+1)
        else:
            y_arr = y_arr.T             # (d, nmax+1)
        return np.vstack([t_arr[np.newaxis], y_arr])
