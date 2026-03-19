"""
Computational Physics
Directory: src/calculus/schemes

Code: First-order derivative schemes
    Forward, Backward, Central, Richardson

Author: Barlev Raymond
"""
import numpy as np
from compPhyx.calculus.schemes._base import DerivativeScheme


class ForwardD1(DerivativeScheme):
    """First derivative via forward difference: (f(x+h) - f(x)) / h"""

    def __call__(self, f, x):
        return (f(x + self.h) - f(x)) / self.h

    def differentiate(self, t, y):
        n = len(t) - 1
        d = np.zeros(n + 1)
        d[:-1] = (y[1:] - y[:-1]) / (t[1:] - t[:-1])
        d[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
        return d


class BackwardD1(DerivativeScheme):
    """First derivative via backward difference: (f(x) - f(x-h)) / h"""

    def __call__(self, f, x):
        return (f(x) - f(x - self.h)) / self.h

    def differentiate(self, t, y):
        n = len(t) - 1
        d = np.zeros(n + 1)
        d[0] = (y[1] - y[0]) / (t[1] - t[0])
        d[1:] = (y[1:] - y[:-1]) / (t[1:] - t[:-1])
        return d


class CentralD1(DerivativeScheme):
    """First derivative via central difference: (f(x+h) - f(x-h)) / (2h)"""

    def __call__(self, f, x):
        return (f(x + self.h) - f(x - self.h)) / (2 * self.h)

    def differentiate(self, t, y):
        n = len(t) - 1
        d = np.zeros(n + 1)
        d[0] = (y[1] - y[0]) / (t[1] - t[0])
        d[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
        d[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
        return d


class RichardsonD1(DerivativeScheme):
    """
    First derivative via Richardson extrapolation (4th-order accurate):
        (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    """

    def __call__(self, f, x):
        return (f(x - 2*self.h) - 8*f(x - self.h) + 8*f(x + self.h) - f(x + 2*self.h)) / (12 * self.h)

    def differentiate(self, t, y):
        n = len(t) - 1
        d = np.zeros(n + 1)
        # boundaries fall back to lower-order schemes
        d[0] = (y[1] - y[0]) / (t[1] - t[0])
        d[1] = (y[2] - y[0]) / (t[2] - t[0])
        d[2:-2] = (-y[4:] + 8*y[3:-1] - 8*y[1:-3] + y[:-4]) / (12 * (t[3:-1] - t[2:-2]))
        d[-2] = (y[-1] - y[-3]) / (t[-1] - t[-3])
        d[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
        return d
