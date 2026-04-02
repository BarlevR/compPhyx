'''
compPhyx.applications._base

Base class for ODE-based physics applications.
Provides a shared solve() that dispatches to any compPhyx fixed-step solver.
Subclasses implement f(t, r) and set self.r0 in __init__.
'''

from types import SimpleNamespace
from compPhyx.timestepping import Euler, RK4, RK45

_SOLVERS = {
    'Euler': Euler,
    'RK4':   RK4,
    'RK45':  RK45,
}


class Application:

    def solve(self, method='RK45', t_span=None, t_eval=None):
        '''
        Integrate the application's ODE from t_span[0] to t_span[1].

        Parameters
        ----------
        method : str, one of 'Euler', 'RK4', 'RK45'
        t_span : (t0, tEnd)
        t_eval : array_like, uniformly spaced evaluation times

        Returns
        -------
        SimpleNamespace with:
            .t  — shape (nmax+1,)
            .y  — shape (d, nmax+1), one row per state component
        '''
        if method not in _SOLVERS:
            raise ValueError(f"method must be one of {list(_SOLVERS)}. Got '{method}'.")
        t0, tEnd = t_span
        nmax = len(t_eval) - 1
        h    = (tEnd - t0) / nmax
        raw  = _SOLVERS[method](self.f, t0, self.r0, nmax, h).solve()
        return SimpleNamespace(t=raw[0], y=raw[1:])
