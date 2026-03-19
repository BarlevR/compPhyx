"""
Computational Physics
Directory: src/calculus/schemes

Code: Second-order derivative schemes
    Forward, Backward, Central, Richardson

Author: Barlev Raymond
"""
import numpy as np
from compPhyx.calculus.schemes._base import DerivativeScheme


class ForwardD2(DerivativeScheme):
    """Second derivative via forward difference: (f(x+2h) - 2f(x+h) + f(x)) / h^2"""

    def __call__(self, f, x):
        return (f(x + 2*self.h) - 2*f(x + self.h) + f(x)) / self.h**2

    def differentiate(self, t, y):
        n = len(t) - 1
        d = np.zeros(n + 1)
        for i in range(n - 1):
            d[i] = ((y[i+2] - y[i+1]) / (t[i+2] - t[i+1]) - (y[i+1] - y[i]) / (t[i+1] - t[i])) / (t[i+1] - t[i])
        d[n-1] = ((y[n-1] - y[n-2]) / (t[n-1] - t[n-2]) - (y[n-2] - y[n-3]) / (t[n-2] - t[n-3])) / (t[n-1] - t[n-2])
        d[n] = ((y[n] - y[n-1]) / (t[n] - t[n-1]) - (y[n-1] - y[n-2]) / (t[n-1] - t[n-2])) / (t[n] - t[n-1])
        return d


class BackwardD2(DerivativeScheme):
    """Second derivative via backward difference: (f(x) - 2f(x-h) + f(x-2h)) / h^2"""

    def __call__(self, f, x):
        return (f(x) - 2*f(x - self.h) + f(x - 2*self.h)) / self.h**2

    def differentiate(self, t, y):
        n = len(t) - 1
        d = np.zeros(n + 1)
        d[0] = ((y[2] - y[1]) / (t[2] - t[1]) - (y[1] - y[0]) / (t[1] - t[0])) / (t[1] - t[0])
        d[1] = ((y[2] - y[1]) / (t[2] - t[1]) - (y[1] - y[0]) / (t[1] - t[0])) / (t[1] - t[0])
        for i in range(2, n + 1):
            d[i] = ((y[i] - y[i-1]) / (t[i] - t[i-1]) - (y[i-1] - y[i-2]) / (t[i-1] - t[i-2])) / (t[i] - t[i-1])
        return d


class CentralD2(DerivativeScheme):
    """Second derivative via central difference: (f(x+h) - 2f(x) + f(x-h)) / h^2"""

    def __call__(self, f, x):
        return (f(x + self.h) - 2*f(x) + f(x - self.h)) / self.h**2

    def differentiate(self, t, y):
        n = len(t) - 1
        d = np.zeros(n + 1)
        d[0] = ((y[2] - y[1]) / (t[2] - t[1]) - (y[1] - y[0]) / (t[1] - t[0])) / (t[1] - t[0])
        for i in range(1, n):
            d[i] = (y[i+1] - 2*y[i] + y[i-1]) / ((t[i+1] - t[i]) * (t[i] - t[i-1]))
        d[n] = ((y[n] - y[n-1]) / (t[n] - t[n-1]) - (y[n-1] - y[n-2]) / (t[n-1] - t[n-2])) / (t[n] - t[n-1])
        return d


class RichardsonD2(DerivativeScheme):
    """
    Second derivative via Richardson extrapolation (4th-order accurate):
        (-f(x-2h) + 16f(x-h) - 30f(x) + 16f(x+h) - f(x+2h)) / (12h^2)
    """

    def __call__(self, f, x):
        return (-f(x - 2*self.h) + 16*f(x - self.h) - 30*f(x) + 16*f(x + self.h) - f(x + 2*self.h)) / (12 * self.h**2)

    def differentiate(self, t, y):
        n = len(t) - 1
        d = np.zeros(n + 1)
        # boundaries fall back to central scheme
        d[0] = ((y[2] - y[1]) / (t[2] - t[1]) - (y[1] - y[0]) / (t[1] - t[0])) / (t[1] - t[0])
        d[1] = (y[2] - 2*y[1] + y[0]) / ((t[2] - t[1]) * (t[1] - t[0]))
        for i in range(2, n - 1):
            d[i] = (-y[i-2] + 16*y[i-1] - 30*y[i] + 16*y[i+1] - y[i+2]) / (12 * (t[i+1] - t[i]) * (t[i] - t[i-1]))
        d[n-1] = (y[n] - 2*y[n-1] + y[n-2]) / ((t[n] - t[n-1]) * (t[n-1] - t[n-2]))
        d[n] = ((y[n] - y[n-1]) / (t[n] - t[n-1]) - (y[n-1] - y[n-2]) / (t[n-1] - t[n-2])) / (t[n] - t[n-1])
        return d
