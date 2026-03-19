"""
Computational Physics
Directory: src/calculus/schemes

Code: Abstract base class for derivative schemes

Author: Barlev Raymond
"""
from abc import ABC, abstractmethod
import numpy as np


class DerivativeScheme(ABC):
    """
    Abstract base class for numerical derivative schemes.

    Subclasses implement __call__ for function-based differentiation
    and differentiate for array-based (data) differentiation.
    """

    def __init__(self, h=1e-5):
        if h <= 0:
            raise ValueError("h must be > 0")
        self.h = h

    @abstractmethod
    def __call__(self, f, x):
        """
        Compute the derivative of f at x.

        Parameters
        ----------
        f : callable
        x : float or np.ndarray

        Returns
        -------
        float or np.ndarray
        """

    @abstractmethod
    def differentiate(self, t, y):
        """
        Compute the derivative from discrete (t, y) data arrays.
        Handles non-uniform spacing. Boundary points fall back to
        forward/backward differences as appropriate.

        Parameters
        ----------
        t : np.ndarray, shape (N,)
        y : np.ndarray, shape (N,)

        Returns
        -------
        np.ndarray, shape (N,)
        """
