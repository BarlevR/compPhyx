'''
compPhyx.applications.heatEquation2D

2D heat equation solved via the Method of Lines.

The PDE:
    ∂u/∂t = a * (∂²u/∂x² + ∂²u/∂y²)

is discretised in space on an (nx × ny) uniform grid using finite
differences, reducing it to a system of nx*ny ODEs integrated forward
in time.  The state vector is flattened (row-major) for the ODE solver
and reshaped internally at each function evaluation.

Dirichlet boundary conditions are enforced on all four edges by zeroing
the time derivatives at boundary nodes.

Author: Barlev Raymond
'''

import numpy as np
from ._base import Application
from compPhyx.calculus.schemes import CentralLaplacian, RichardsonLaplacian

_SPATIAL_SCHEMES = {
    'CentralLaplacian':    CentralLaplacian,
    'RichardsonLaplacian': RichardsonLaplacian,
}


class HeatEquation2D(Application):
    '''
    Parameters
    ----------
    thermal_diffusivity : float, thermal diffusivity coefficient (a)
    dx                  : float, spatial step size in x
    dy                  : float, spatial step size in y  (defaults to dx)
    nx                  : int,   number of grid points in x
    ny                  : int,   number of grid points in y
    bc_left             : float, fixed temperature at x=0 edge
    bc_right            : float, fixed temperature at x=L edge
    bc_bottom           : float, fixed temperature at y=0 edge
    bc_top              : float, fixed temperature at y=H edge
    u0                  : array_like of shape (nx, ny), or None
                          Initial temperature field. If None, defaults to
                          zeros with boundary conditions applied.
    spatial_scheme      : str, spatial Laplacian scheme.
                          One of 'CentralLaplacian' (2nd-order) or
                          'RichardsonLaplacian' (4th-order).
    '''

    def __init__(self, thermal_diffusivity=1.0, dx=1.0, dy=None,
                 nx=50, ny=50,
                 bc_left=0.0, bc_right=0.0, bc_bottom=0.0, bc_top=0.0,
                 u0=None, spatial_scheme='CentralLaplacian'):
        self.a  = thermal_diffusivity
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.nx = nx
        self.ny = ny

        if u0 is not None:
            u = np.asarray(u0, dtype=float).copy()
        else:
            u = np.zeros((nx, ny))

        # Enforce Dirichlet boundary conditions on all four edges
        u[0,  :] = bc_left
        u[-1, :] = bc_right
        u[:,  0] = bc_bottom
        u[:, -1] = bc_top

        # Flatten to 1D for the ODE solver
        self.r0 = u.ravel()

        if spatial_scheme not in _SPATIAL_SCHEMES:
            available = ', '.join(_SPATIAL_SCHEMES.keys())
            raise ValueError(
                f"Unknown spatial_scheme '{spatial_scheme}'. "
                f"Available schemes are: {available}."
            )

        self._laplacian  = _SPATIAL_SCHEMES[spatial_scheme](dx=dx, dy=self.dy)
        self.spatial_scheme = spatial_scheme

        # Store BC values for re-enforcement each step
        self._bc = dict(left=bc_left, right=bc_right,
                        bottom=bc_bottom, top=bc_top)

    def f(self, t, u_flat):
        '''2D finite difference Laplacian — ODE right-hand side.'''
        u   = u_flat.reshape(self.nx, self.ny)
        d2u = self._laplacian.differentiate(u)

        # Dirichlet BCs: zero time derivatives on all four edges
        d2u[0,  :] = 0.0
        d2u[-1, :] = 0.0
        d2u[:,  0] = 0.0
        d2u[:, -1] = 0.0

        return (self.a * d2u).ravel()

    def reshape(self, y_flat):
        '''Reshape a flat solution row back to (nx, ny).'''
        return y_flat.reshape(self.nx, self.ny)
