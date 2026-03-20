"""
Computational Physics
Directory: src/calculus/integration

Code: Numerical integration schemes
    Sum, Trapezoidal, Simpson

Author: Barlev Raymond
"""
import numpy as np
from compPhyx.calculus.integration._base import IntegrationScheme


class Sum(IntegrationScheme):
    """
    Rectangle (sum) rule: assumes uniform spacing.
        integral ≈ sum(y) * (x[-1] - x[0]) / (N - 1)
    """

    def __call__(self, x, y):
        return np.sum(y) * (x[-1] - x[0]) / (len(y) - 1)


class Trapezoidal(IntegrationScheme):
    """
    Trapezoidal rule: works with non-uniform spacing.
        integral ≈ sum((y[i+1] + y[i]) / 2 * (x[i+1] - x[i]))
    """

    def __call__(self, x, y):
        return np.sum((y[1:] + y[:-1]) / 2 * (x[1:] - x[:-1]))


class Simpson(IntegrationScheme):
    """
    Simpson's rule: assumes uniform spacing, requires odd number of points.
        Uses alternating 4/3 and 2/3 coefficients.
    """

    def __call__(self, x, y):
        dx = (x[-1] - x[0]) / (len(y) - 1)
        return (1/3*y[0] + 4/3*np.sum(y[1:-1:2]) + 2/3*np.sum(y[2:-1:2]) + 1/3*y[-1]) * dx
