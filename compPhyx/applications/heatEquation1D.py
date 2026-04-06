'''
compPhyx.applications.heatEquation1D

1D heat equation solved via the Method of Lines.

The PDE:
    ∂u/∂t = a * ∂²u/∂x²

is discretised in space using finite differences, reducing it to
a system of ODEs — one per grid point:

    du_i/dt = a/dx² * (u_{i+1} - 2*u_i + u_{i-1})

Dirichlet boundary conditions are enforced by fixing the first and
last grid points: their time derivatives are always zero so they
never change from their initial values.

Author: Barlev Raymond
'''

import numpy as np
from ._base import Application
from compPhyx.calculus.schemes import CentralD2


class HeatEquation1D(Application):
    '''
    Parameters
    ----------
    thermal_diffusivity : float, thermal diffusivity coefficient (a)
    dx                  : float, spatial step size
    n_points            : int,   number of spatial grid points
    bc_left             : float, fixed temperature at left boundary
    bc_right            : float, fixed temperature at right boundary
    u0                  : array_like or None
                          Initial temperature profile. If None, defaults
                          to zeros with boundary conditions applied.
    '''

    def __init__(self, thermal_diffusivity=1.0, dx=1.0, n_points=100,
                 bc_left=0.0, bc_right=0.0, u0=None):
        self.a  = thermal_diffusivity
        self.dx = dx
        self.n  = n_points

        if u0 is not None:
            self.r0 = np.asarray(u0, dtype=float).copy()
        else:
            self.r0 = np.zeros(n_points)

        # Enforce Dirichlet boundary conditions
        self.r0[0]  = bc_left
        self.r0[-1] = bc_right

        # Spatial grid and second-derivative scheme
        self._x_grid = np.arange(n_points, dtype=float) * dx
        self._d2     = CentralD2(h=dx)

    def f(self, t, u):
        '''Finite difference Laplacian — ODE right-hand side.'''
        d2u = self._d2.differentiate(self._x_grid, u)
        d2u[0]  = 0.0   # Dirichlet BC: left boundary fixed
        d2u[-1] = 0.0   # Dirichlet BC: right boundary fixed
        return self.a * d2u
