"""
compPhyx.calculus.schemes.laplacian

Laplacian operator schemes for 1D and 2D uniform grids.

Each class stores the grid spacing at construction and exposes a single
`differentiate(u)` method that dispatches on `u.ndim`:

    u.ndim == 1  →  d²u/dx²          (1D second derivative)
    u.ndim == 2  →  d²u/dx² + d²u/dy²  (2D 5-point / 9-point Laplacian)

Boundary rows/columns are set to zero — Dirichlet boundary conditions are
enforced by the application layer, not the scheme.

Author: Barlev Raymond
"""
from abc import ABC, abstractmethod
import numpy as np


class LaplaceScheme(ABC):
    """
    Abstract base class for Laplacian schemes on uniform grids.

    Parameters
    ----------
    dx : float, grid spacing in x (and y if dy is not supplied)
    dy : float or None, grid spacing in y  (2D only; defaults to dx)
    """

    def __init__(self, dx, dy=None):
        if dx <= 0:
            raise ValueError("dx must be > 0")
        self.dx = float(dx)
        self.dy = float(dy) if dy is not None else float(dx)

    @abstractmethod
    def differentiate(self, u):
        """
        Apply the Laplacian stencil to grid values u.

        Parameters
        ----------
        u : np.ndarray, shape (N,) for 1D or (Nx, Ny) for 2D

        Returns
        -------
        np.ndarray, same shape as u
        """


class CentralLaplacian(LaplaceScheme):
    """
    2nd-order central-difference Laplacian.

    1D stencil:  (u[i-1] - 2u[i] + u[i+1]) / dx²
    2D stencil:  (u[i-1,j] - 2u[i,j] + u[i+1,j]) / dx²
               + (u[i,j-1] - 2u[i,j] + u[i,j+1]) / dy²
    """

    def differentiate(self, u):
        if u.ndim == 1:
            return self._apply_1d(u)
        elif u.ndim == 2:
            return self._apply_2d(u)
        raise ValueError(f"CentralLaplacian supports 1D or 2D arrays, got ndim={u.ndim}")

    def _apply_1d(self, u):
        d = np.zeros_like(u)
        d[1:-1] = (u[:-2] - 2*u[1:-1] + u[2:]) / self.dx**2
        return d

    def _apply_2d(self, u):
        d = np.zeros_like(u)
        d[1:-1, 1:-1] = (
            (u[:-2, 1:-1] - 2*u[1:-1, 1:-1] + u[2:,  1:-1]) / self.dx**2 +
            (u[1:-1, :-2] - 2*u[1:-1, 1:-1] + u[1:-1, 2: ]) / self.dy**2
        )
        return d


class RichardsonLaplacian(LaplaceScheme):
    """
    4th-order Richardson-extrapolated Laplacian.

    1D stencil (interior):
        (-u[i-2] + 16u[i-1] - 30u[i] + 16u[i+1] - u[i+2]) / (12 dx²)
    Points at distance 1 from each boundary fall back to CentralLaplacian.

    2D stencil (interior): same formula applied independently in x and y.
    Near-boundary rows/columns fall back to the central stencil in the
    direction that cannot reach two points away.
    """

    def differentiate(self, u):
        if u.ndim == 1:
            return self._apply_1d(u)
        elif u.ndim == 2:
            return self._apply_2d(u)
        raise ValueError(f"RichardsonLaplacian supports 1D or 2D arrays, got ndim={u.ndim}")

    def _apply_1d(self, u):
        d = np.zeros_like(u)
        # Richardson interior
        d[2:-2] = (-u[:-4] + 16*u[1:-3] - 30*u[2:-2] + 16*u[3:-1] - u[4:]) / (12 * self.dx**2)
        # Central fallback at i=1 and i=n-2
        if len(u) > 4:
            d[1]  = (u[0]  - 2*u[1]  + u[2])  / self.dx**2
            d[-2] = (u[-3] - 2*u[-2] + u[-1]) / self.dx**2
        return d

    def _apply_2d(self, u):
        d = np.zeros_like(u)
        nx, ny = u.shape

        # --- full Richardson interior (need 2 neighbours each side) ---
        if nx > 4 and ny > 4:
            d[2:-2, 2:-2] = (
                (-u[:-4,  2:-2] + 16*u[1:-3, 2:-2] - 30*u[2:-2, 2:-2] + 16*u[3:-1, 2:-2] - u[4:,  2:-2]) / (12 * self.dx**2) +
                (-u[2:-2, :-4 ] + 16*u[2:-2, 1:-3] - 30*u[2:-2, 2:-2] + 16*u[2:-2, 3:-1] - u[2:-2, 4: ]) / (12 * self.dy**2)
            )

        # --- near-boundary rows (i=1 and i=nx-2): central in x, Richardson in y ---
        for i in [1, nx - 2]:
            if 0 < i < nx - 1 and ny > 4:
                d[i, 2:-2] = (
                    (u[i-1, 2:-2] - 2*u[i, 2:-2] + u[i+1, 2:-2]) / self.dx**2 +
                    (-u[i, :-4] + 16*u[i, 1:-3] - 30*u[i, 2:-2] + 16*u[i, 3:-1] - u[i, 4:]) / (12 * self.dy**2)
                )
                # corner columns of these rows: full central
                if ny > 2:
                    d[i, 1]  = (u[i-1, 1]  - 2*u[i, 1]  + u[i+1, 1])  / self.dx**2 + (u[i, 0]  - 2*u[i, 1]  + u[i, 2])  / self.dy**2
                    d[i, -2] = (u[i-1, -2] - 2*u[i, -2] + u[i+1, -2]) / self.dx**2 + (u[i, -3] - 2*u[i, -2] + u[i, -1]) / self.dy**2

        # --- near-boundary columns (j=1 and j=ny-2): Richardson in x, central in y ---
        for j in [1, ny - 2]:
            if 0 < j < ny - 1 and nx > 4:
                d[2:-2, j] = (
                    (-u[:-4, j] + 16*u[1:-3, j] - 30*u[2:-2, j] + 16*u[3:-1, j] - u[4:, j]) / (12 * self.dx**2) +
                    (u[2:-2, j-1] - 2*u[2:-2, j] + u[2:-2, j+1]) / self.dy**2
                )

        return d
