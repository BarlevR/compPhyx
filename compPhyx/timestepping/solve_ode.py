'''
compPhyx.timestepping.solve_ode

Unified interface for compPhyx ODE solvers, mirroring scipy's solve_ivp
return structure so examples can swap methods with a single variable.
'''

from types import SimpleNamespace
import numpy as np
from .euler import Euler
from .rungeKutta import RK4, RK45

_SOLVERS = {
    'Euler': Euler,
    'RK4':   RK4,
    'RK45':  RK45,
}

def solve_ode(f, t_span, y0, method='RK45', t_eval=None):
    '''
    Solve an ODE system dy/dt = f(t, y) using a compPhyx fixed-step solver.

    Parameters
    ----------
    f      : callable, f(t, y) — ODE right-hand side
    t_span : (t0, tEnd)
    y0     : scalar or array_like, initial condition
    method : str, one of 'Euler', 'RK4', 'RK45'
    t_eval : array_like, uniformly spaced evaluation times
             (used to derive step size h and nmax)

    Returns
    -------
    SimpleNamespace with:
        .t  — np.ndarray, shape (nmax+1,)
        .y  — np.ndarray, shape (d, nmax+1)  — one row per state component
    '''
    if method not in _SOLVERS:
        raise ValueError(f"method must be one of {list(_SOLVERS)}. Got '{method}'.")

    t0, tEnd = t_span
    if t_eval is not None:
        nmax = len(t_eval) - 1
    else:
        raise ValueError("t_eval is required.")
    h = (tEnd - t0) / nmax

    raw = _SOLVERS[method](f, t0, y0, nmax, h).solve()
    # raw[0] = t, raw[1:] = state components
    return SimpleNamespace(t=raw[0], y=raw[1:])
