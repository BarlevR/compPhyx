'''
compPhyx.timestepping._base

Abstract base class for fixed-step ODE solvers.
All solvers share the same __init__ and result-packing logic.
Subclasses implement only the stepping logic in solve().

Author: Barlev Raymond
'''

from abc import ABC, abstractmethod
import numpy as np


class _ODESolver(ABC):

    def __init__(self, f, t0, y0, nmax, h):
        self.f    = f
        self.t0   = float(t0)
        self.y0   = np.asarray(y0, dtype=float)
        self.nmax = int(nmax)
        self.h    = float(h)

    @abstractmethod
    def solve(self):
        '''
        Integrate the ODE and return the solution array.

        Returns
        -------
        np.ndarray, shape (d+1, nmax+1)
            Row 0 is t. Rows 1..d are state components.
            For scalar y0, shape is (2, nmax+1).
        '''

    def _pack_result(self, t_list, y_list):
        '''Convert accumulated step lists into the standard return array.'''
        t_arr = np.array(t_list)
        y_arr = np.array(y_list)
        if y_arr.ndim == 1:
            y_arr = y_arr[np.newaxis]   # (1, nmax+1)
        else:
            y_arr = y_arr.T             # (d, nmax+1)
        return np.vstack([t_arr[np.newaxis], y_arr])
