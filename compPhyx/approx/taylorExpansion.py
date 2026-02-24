"""
Computational Physics
Directory: src/approx/taylorExpansion

Code: calculate the taylor expansion of a general function

Author: Barlev Raymond
"""
# Imports

import math
import numpy as np
from compPhyx.calculus.numericalDerivative import nth_derivative

# Define the class
class TaylorSeries:
    """
    Taylor series approximation of a function f about x0 up to order nmax.
    Uses a numerical nth-derivative function (default: forward finite difference).
    """

    def __init__(self, f, x0, nmax, h=1e-3, derivative_fn=nth_derivative):
        self.f = f
        self.x0 = float(x0)
        self.nmax = int(nmax)
        self.h = float(h)
        self.derivative_fn = derivative_fn

        if self.nmax < 0:
            raise ValueError("nmax must be >= 0")
        if self.h <= 0:
            raise ValueError("h must be > 0")

    def evaluate(self, x):
        """
        Evaluate the Taylor polynomial at x.
        x can be a float or a numpy array.
        """
        x_arr = np.asarray(x) if not np.isscalar(x) else x
        dx = x_arr - self.x0

        total = 0.0
        for n in range(self.nmax + 1):
            dn = self.derivative_fn(f=self.f, x=self.x0, n=n, h=self.h)  # f^(n)(x0)
            total += dn * (dx ** n) / math.factorial(n)

        return total
