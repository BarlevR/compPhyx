'''
compPhyx.timestepping.rungeKutta

Runge-Kutta ODE solvers.

  RK4   — classical 4th-order Runge-Kutta
  RK45  — Runge-Kutta-Fehlberg 5th-order method
'''

from ._base import _ODESolver


class RK4(_ODESolver):
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

    def solve(self):
        t = self.t0
        y = self.y0
        t_list = [t]
        y_list = [y]
        for _ in range(self.nmax):
            k1 = self.h * self.f(t, y)
            k2 = self.h * self.f(t + self.h/2, y + k1/2)
            k3 = self.h * self.f(t + self.h/2, y + k2/2)
            k4 = self.h * self.f(t + self.h,   y + k3)
            y  = y + (k1 + 2*k2 + 2*k3 + k4) / 6
            t  = t + self.h
            t_list.append(t)
            y_list.append(y)
        return self._pack_result(t_list, y_list)


class RK45(_ODESolver):
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

    def solve(self):
        t = self.t0
        y = self.y0
        t_list = [t]
        y_list = [y]
        for _ in range(self.nmax):
            k1 = self.h * self.f(t, y)
            k2 = self.h * self.f(t + self.h/4,      y + k1/4)
            k3 = self.h * self.f(t + 3*self.h/8,    y + 3*k1/32      + 9*k2/32)
            k4 = self.h * self.f(t + 12*self.h/13,  y + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
            k5 = self.h * self.f(t + self.h,         y + 439*k1/216   - 8*k2         + 3680*k3/513  - 845*k4/4104)
            k6 = self.h * self.f(t + self.h/2,       y - 8*k1/27      + 2*k2         - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)
            y  = y + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55
            t  = t + self.h
            t_list.append(t)
            y_list.append(y)
        return self._pack_result(t_list, y_list)
